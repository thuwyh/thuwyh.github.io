---
categories:
- Deep Learning
comments: true
date: "2021-06-20T10:25:03Z"
header:
  teaser: /assets/light.jpeg
tags:
- Inference
title: Impvove Inference Efficiency with Batch Inference
toc: true
---

As an algorithm engineer, it is inevitable that you will encounter the problem of bringing models online in your daily work. For some less demanding scenarios, you can handle this by utilizing a web framework: for each user request, call the model to infer and return the result. However, this straightforward implementation often fails to maximize the use of the GPU, and is slightly overwhelming for scenarios with high performance requirements.

There are many ways to optimize, and one useful tip is to change from inference for each request to inference for multiple requests at once. Last year, about this time I wrote a small tool to achieve this function and gave it a rather overbearing name InferLight. Honestly,  that tool was not very well implemented. Recently, I refactor the tool with reference to Shannon Technology's Service-Streamer . 

This feature seems simple, but in the process of implementation, we can understand a lot of Python asynchronous programming knowledge and feel the parallel computing power of modern GPU.

## Architecture
First, to improve the model's online inference throughput, you should make the inference service asynchronous. For web services, asynchronous means that the program can handle other requests while the model is computing. For Python, asynchronous services can be implemented with good Asyncio-based frameworks, such as Sanic , which I commonly use. Whereas inference is computationally intensive, our goal is to be able to aggregate multiple inference requests, make efficient use of the parallel computing power of the GPU, and be able to return the results of bulk inference to the corresponding requestor correctly.

To achieve the above goal, the following modules are needed

1.	Front-end service: used to receive requests and return results. It can be various protocols such as Http, PRC, etc. It is an independent process.
2.	Inference Worker: responsible for model initialization, bulk inference data construction, and inference calculation. It is an independent process.
3.	Task queue: the front-end service receives the request and sends the calculation task to the task queue; the inference worker listens to the queue and takes out a small batch each time by the model inference
4.	Result queue: After the inference done, inference worker sends the result to the result queue; the front-end service listens to the queue and gets the inference result
5.	Result distribution: before sending the task to the task queue, a unique identifier of the task needs to be generated, and the result corresponding to the task is obtained according to the identifier after retrieving the result from the result queue

There are many ways to implement the task queue and result queue, and you can use some mature middleware such as Kafka and Redis. To avoid external dependencies, I chose to use Python's native multi-process queue this time. The result queue is listened to and distributed through a sub-thread of the front-end service process.

## Implementation
The inference worker is relatively simple. Since there are a variety of models to load and data processing steps, I designed the inference worker as a base class that is inherited and implements specific methods when used.

```python
class BaseInferLightWorker:

    def __init__(self, data_queue:mp.Queue, result_queue:mp.Queue, 
                 model_args:dict, 
                 batch_size=16, max_delay=0.1,
                 ready_event=None) -> None:
        self.data_queue = data_queue
        self.result_queue = result_queue
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.logger = logging.getLogger('InferLight-Worker')
        self.logger.setLevel(logging.DEBUG)

        self.load_model(model_args)
        
        # Inform parent process when model loaded
        if ready_event:
            ready_event.set()

    def run(self):
        self.logger.info('Worker started!')
        while True:
            data, task_ids = [], []
            since = time.time()
            for i in range(self.batch_size):
                try:
                    # get data form data queue
                    d = self.data_queue.get(block=True, timeout=self.max_delay)
                    task_ids.append(d[0])
                    data.append(d[1])
                    self.logger.info('get one new task')
                except Empty:
                    pass
                if time.time()-since>=self.max_delay:
                    break
            if len(data)>0:
                start = time.perf_counter()
                batch = self.build_batch(data)
                results = self.inference(batch)
                end = time.perf_counter()
                time_elapsed = (end-start)*1000
                self.logger.info(f'inference succeeded. batch size: {len(data)}, time elapsed: {time_elapsed:.3f} ms')
                # write results to result queue
                for (task_id, result) in zip(task_ids, results):
                    self.result_queue.put((task_id, result))


    def build_batch(self, requests):
        raise NotImplementedError

    def inference(self, batch):
        raise NotImplementedError

    def load_model(self, model_args):
        raise NotImplementedError

    @classmethod
    def start(cls, data_queue:mp.Queue, result_queue:mp.Queue, model_args:dict, batch_size=16, max_delay=0.1,ready_event=None):
        w = cls(data_queue, result_queue, model_args, batch_size, max_delay, ready_event)
        w.run()
```

Along with this is a Wrapper class used in the front-end service to do the request receiving, result collection and distribution of inference requests.

```python
import asyncio
import logging
import multiprocessing as mp
import threading
import uuid
from queue import Empty

from cachetools import TTLCache

from .data import InferStatus, InferResponse


class LightWrapper:

    def __init__(self, worker_class, model_args: dict,
                 batch_size=16, max_delay=0.1) -> None:
        # setup logger
        self.logger = logging.getLogger('InferLight-Wrapper')
        self.logger.setLevel(logging.INFO)
        
        # save results in a TTL cache
        self.result_cache = TTLCache(maxsize=10000, ttl=5)

        self.mp = mp.get_context('spawn')
        self.result_queue = self.mp.Queue()
        self.data_queue = self.mp.Queue()

        # start inference worker process
        self.logger.info('Starting worker...')
        worker_ready_event = self.mp.Event()
        self._worker_p = self.mp.Process(target=worker_class.start, args=(
            self.data_queue, self.result_queue, model_args, batch_size, max_delay, worker_ready_event
        ), daemon=True)
        self._worker_p.start()
        
        # wait at most 30 seconds
        is_ready = worker_ready_event.wait(timeout=30)
        if is_ready:
            self.logger.info('Worker started!')
        else:
            self.logger.error('Failed to start worker!')
        
        # start the result collecting thread
        self.back_thread = threading.Thread(
            target=self._collect_result, name="thread_collect_result")
        self.back_thread.daemon = True
        self.back_thread.start()

    def _collect_result(self):
        # keep reading result queue
        # write result to cache with task_id as key
        self.logger.info('Result collecting thread started!')
        while True:
            try:
                msg = self.result_queue.get(block=True, timeout=0.01)
            except Empty:
                msg = None
            if msg is not None:
                (task_id, result) = msg
                self.result_cache[task_id] = result

    async def get_result(self, task_id):
        # non-blocking check result
        while task_id not in self.result_cache:
            await asyncio.sleep(0.01)
        return self.result_cache[task_id]

    async def predict(self, input, timeout=2) -> InferResponse:
        # generate unique task_id
        task_id = str(uuid.uuid4())

        # send input to worker process
        self.data_queue.put((task_id, input))
        try:
            # here we set a timeout threshold to avoid waiting forever
            result = await asyncio.wait_for(self.get_result(task_id), timeout=timeout)
        except asyncio.TimeoutError:
            return InferResponse(InferStatus.TIMEOUT, None)

        return InferResponse(InferStatus.SUCCEED, result)
```

Some of the data structures used are defined as follows

```python
from enum import Enum

class InferStatus(Enum):
  SUCCEED = 0
  TIMEOUT = 1

class InferResponse:

  def __init__(self, status: InferStatus, result) -> None:
      self.status = status
      self.result = result

  def succeed(self):
      return self.status==InferStatus.SUCCEED
```

## Use Case and Test Result
Here we show how the above components can be used with a sentiment analysis BERT model.

First define the model

```python
class BertModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.bert = AutoModelForSequenceClassification.from_pretrained(config['model'])
        self.bert.eval()
        self.device = torch.device('cuda' if config.get('use_cuda') else 'cpu')
        self.bert.to(self.device)

    def forward(self, inputs):
        return self.bert(**inputs).logits
```

Then inherit BaseInferLightWorker and implement three functions to get a complete Worker class

```python
class MyWorker(BaseInferLightWorker):

    def load_model(self, model_args):
        self.model = BertModel(model_args)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_args['model'])
        self.device = torch.device('cuda' if model_args.get('use_cuda') else 'cpu')
        return

    def build_batch(self, requests):
        # 这个函数用来构建batch inference的输入
        encoded_input = self.tokenizer.batch_encode_plus(requests, 
                                                         return_tensors='pt',
                                                         padding=True,
                                                         truncation=True,
                                                         max_length=512)
        return encoded_input.to(self.device)

    @torch.no_grad()
    def inference(self, batch):
        model_output = self.model.forward(batch).cpu().numpy()
        scores = softmax(model_output, axis=1)
        # 将整个batch的结果以list形式返回即可
        ret = [x.tolist() for x in scores]
        return ret
```

Finally, building services

```python
if __name__=='__main__':
    # for convenience，we use a fixed text from Aesop's Fables as input
    text = """
    A Fox one day spied a beautiful bunch of ripe grapes hanging from a vine trained along the branches of a tree. The grapes seemed ready to burst with juice, and the Fox's mouth watered as he gazed longingly at them.
    The bunch hung from a high branch, and the Fox had to jump for it. The first time he jumped he missed it by a long way. So he walked off a short distance and took a running leap at it, only to fall short once more. Again and again he tried, but in vain.
    Now he sat down and looked at the grapes in disgust.
    "What a fool I am," he said. "Here I am wearing myself out to get a bunch of sour grapes that are not worth gaping for."
    And off he walked very, very scornfully.
    """
    
    config = {
        'model':"nlptown/bert-base-multilingual-uncased-sentiment",
        'use_cuda':True
    }
    wrapped_model = LightWrapper(MyWorker, config, batch_size=16, max_delay=0.05)
    
    app = Sanic('test')
    
    @app.get('/batch_predict')
    async def batched_predict(request):
        dummy_input = text
        response = await wrapped_model.predict(dummy_input)
        if not response.succeed():
            return json_response({'output':None, 'status':'failed'})
        return json_response({'output': response.result})

    app.run(port=8888)
```

I did some tests with the famous Apache’s ab tool. I started the above app on my HP Z4 Workstation and made sure the worker process was running on a RTX 6000 GPU.

With `ab -n 1000 -c 32 http://localhost:8888/batched_predict`, I got the following result.

```
Concurrency Level:      32
Time taken for tests:   4.019 seconds
Complete requests:      1000
Failed requests:        999
   (Connect: 0, Receive: 0, Length: 999, Exceptions: 0)
Total transferred:      202978 bytes
HTML transferred:       111978 bytes
Requests per second:    248.79 [#/sec] (mean)
Time per request:       128.620 [ms] (mean)
Time per request:       4.019 [ms] (mean, across all concurrent requests)
Transfer rate:          49.32 [Kbytes/sec] received
```

Test result of another straightford implement without batch inference is as follow:

```
Concurrency Level:      32
Time taken for tests:   10.164 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      202000 bytes
HTML transferred:       111000 bytes
Requests per second:    98.39 [#/sec] (mean)
Time per request:       325.234 [ms] (mean)
Time per request:       10.164 [ms] (mean, across all concurrent requests)
Transfer rate:          19.41 [Kbytes/sec] received
```

As you can see, we got about 2.5 times throughput with batch inference! When doing the benchmark, I also observed that the GPU utilization is much higher with batch inference.

I have opened source the InferLight, and it can be found at [https://github.com/thuwyh/InferLight](https://github.com/thuwyh/InferLight). Hope you love it :)
