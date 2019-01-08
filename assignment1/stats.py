
def calculate_mean(lst):
  total=0
  for element in lst:
    total += element
  mean_val=total/len(lst)
  return mean_val
def calculate_variance(lst):
  
  mean = calculate_mean(lst)

  sum_diffsq = 0

  for element in lst:
    diff = element - mean
    diffsq = np.square(diff)
    sum_diffsq += diffsq

  sum_diffsq = sum_diffsq/(len(lst)-1)
  return sum_diffsq