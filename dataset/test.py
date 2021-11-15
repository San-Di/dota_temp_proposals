import os
from pandarallel import pandarallel
from multiprocessing import Pool, current_process, Queue

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
pandarallel.initialize(nb_workers=30,progress_bar=True,verbose=2,use_memory_fs = False )


NUM_GPUS = 3
PROC_PER_GPU = 2    

queue = Queue()

def foo(filename):
    gpu_id = queue.get()
    try:
        # run processing on GPU <gpu_id>
        ident = current_process().ident
        print('{}: starting process on GPU {}'.format(ident, gpu_id))
        # ... process filename
        print('{}: finished'.format(ident))
    finally:
        queue.put(gpu_id)

# initialize the queue with the GPU ids

queue.put(0)

pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)
files = ['file{}.xyz'.format(x) for x in range(1000)]
for _ in pool.imap_unordered(foo, files):
    pass
pool.close()
pool.join()