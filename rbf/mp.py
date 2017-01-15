''' 
This module provides a parallelized map function 
'''
from multiprocessing import Process, Queue, cpu_count
try:
  import mkl
  _HAS_MKL = True
except ImportError:
  _HAS_MKL = False    

class ParmapError(Exception):
  def __init__(self,errors):
    msg = 'errors were raised when evaluating parmap:\n'
    for i,e in enumerate(errors):
      if e is not None:
        msg += ('    task %s: ' % i) + repr(e) + '\n'

    self.msg = msg

  def __str__(self):
    return self.msg
 

def _f(f,q_in,q_out,q_err):
  while True:
    i,a = q_in.get()
    if i == 'DONE':
      break

    try:
      # append a None to the error queue which indicates that f was 
      # successfully evaluated
      out_entry = f(a)
      err_entry = None 

    except Exception as err:
      # if there is an error append a None to the out queue and append 
      # the error to the error queue. An error will be raised when all 
      # tasks are completed
      print('WARNING: an error was raised for task %s' % i)

      out_entry = None 
      err_entry = err

    q_out.put((i,out_entry))
    q_err.put((i,err_entry))


def parmap(f,args,workers=None):
  '''  
  evaluates [f(a) for a in args] in parallel

  if workers is 0 then the built-in map is used. If workers is greater 
  than one then the parent process spawns that many worker processes to 
  evaluate the map. 
  
  Parameters
  ----------
  f : callable

  a : list
    list of arguments to *f*
    
  workers : int, optional
    number of subprocess to spawn. Defaults to half the available 
    cores plus one

  NOTES
  -----
  If the *mkl* package is installed then this function first sets the 
  maximum number of allowed threads per process to 1. This is to help 
  prevents spawned subprocesses from using multiple cores. The number 
  of allowed threads is reset after all subprocesses have finished.
    
  '''
  if workers is None:
    # default number of processes to have simultaneously running
    workers = cpu_count()//2 + 1

  if workers < 0:
    raise ValueError('number of worker processes must be 0 or greater')
    
  if workers == 0:
    # perform the map on the parent process
    return [f(i) for i in args]

  # attempt to prevent lower level functions from running in parallel
  if _HAS_MKL:
    starting_threads = mkl.get_max_threads()
    mkl.set_num_threads(1)

  # q_in has a max size of 1 so that args is not copied over to 
  # the next process until absolutely necessary
  q_in = Queue(1)
  q_out = Queue()
  # any exceptions found by the child processes are put in this queue 
  # and then raised by the parent
  q_err = Queue()

  # spawn worker processes
  procs = []
  for i in range(workers):
    p = Process(target=_f,args=(f,q_in,q_out,q_err))
    # process is starting and waiting for something to be put on q_in
    p.start()
    procs += [p] 

  submitted_tasks = 0
  for a in args:
    q_in.put((submitted_tasks,a))
    submitted_tasks += 1

  # indicate that nothing else will be added
  for i in range(workers):
    q_in.put(('DONE',None))


  # allocate list of Nones and then fill it in with the results
  val_list = [None for i in range(submitted_tasks)]
  err_list = [None for i in range(submitted_tasks)]
  for i in range(submitted_tasks):
    idx,err = q_err.get()
    err_list[idx] = err
    idx,val = q_out.get()
    val_list[idx] = val

  # terminate all processes
  for p in procs:
    p.join()

  # close queues
  q_in.close()
  q_out.close()
  q_err.close()

  # raise an error if any were found
  if any([e is not None for e in err_list]):
    raise ParmapError(err_list)

  # reset the number of threads to its original value
  if _HAS_MKL:
    mkl.set_num_threads(starting_threads)
    
  return val_list


