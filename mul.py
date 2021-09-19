def multiple(m, n):

	# inserts all elements from n to
	# (m * n)+1 incremented by n.
	a = range(n, (m * n)+1, n)
	
	print(*a)

# driver code
m = 5
n = 73
multiple(m, n)