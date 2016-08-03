#!/usr/bin/env python
import subprocess as sp
import os

files = os.listdir('.')
for f in files:
  if not f.endswith('.py'):
    continue

  if not f.startswith('test'):
    continue

  if __file__ == f:
    continue
    
  sp.call(['python',f])
  
  
