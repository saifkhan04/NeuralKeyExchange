__author_ = "Saif Khan"

from machine import TreeParityMachine
import matplotlib.pyplot as mpl
import numpy as np
import time
import sys

class KeyExchange:
	def __init__(self, k=3, n=4, l=6):
		'''
		Arguments:
		k - The number of hidden neurons
		n - Then number of input neurons connected to each hidden neuron
		l - Defines the range of each weight ({-L, ..., -2, -1, 0, 1, 2, ..., +L })		'''
		self.k = k
		self.n = n
		self.l = l
		self.sync_history = [] # to store the sync score after every update

		self.Alice = TreeParityMachine(k, n, l)
		self.Bob = TreeParityMachine(k, n, l)
		self.Eve = TreeParityMachine(k, n, l)

	#Random number generator
	def random(self):
		return np.random.randint(-l, l + 1, [k, n])

	#Function to evaluate the synchronization score between two machines.
	def sync_score(self, m1, m2):
		return 1.0 - np.average(1.0 * np.abs(m1.W - m2.W)/(2 * l))
	
	def plot(self):
		#Plot graph 
		mpl.plot(self.sync_history)
		mpl.show()		
	
	def run(self, update_rule):
		#Create 3 machines : Alice, Bob and Eve. Eve will try to intercept the communication between
		#Alice and Bob.
		print("Creating machines : k=" + str(k) + ", n=" + str(n) + ", l=" + str(n))
		print("Using " + update_rule + " update rule.")

		#Synchronize weights
		sync = False # Flag to check if weights are sync
		nb_updates = 0 # Update counter
		nb_eve_updates = 0 # To count the number of times eve updated
		start_time = time.time() # Start time

		while(not sync):

			X = self.random() # Create random vector of dimensions [k, n]

			tauA = self.Alice(X) # Get output from Alice
			tauB = self.Bob(X) # Get output from Bob
			tauE = self.Eve(X) # Get output from Eve

			self.Alice.update(tauB, update_rule) # Update Alice with Bob's output
			self.Bob.update(tauA, update_rule) # Update Bob with Alice's output

			#Eve would update only if tauA = tauB = tauE
			if tauA == tauB == tauE:
				self.Eve.update(tauA, update_rule)
				nb_eve_updates += 1

			nb_updates += 1

			score = 100 * self.sync_score(self.Alice, self.Bob) # Calculate the synchronization of the 2 machines

			self.sync_history.append(score) # Add sync score to history, so that we can plot a graph later.

			sys.stdout.write('\r' + "Synchronization = " + str(int(score)) + "%   /  Updates = " + str(nb_updates) + " / Eve's updates = " + str(nb_eve_updates)) 
			if score == 100: # If synchronization score is 100%, set sync flag = True
				sync = True

		end_time = time.time()
		time_taken = end_time - start_time # Calculate time taken

		#Print results
		print ('\nMachines have been synchronized.')
		print ('Time taken = ' + str(time_taken)+ " seconds.")
		print ('Updates = ' + str(nb_updates) + ".")

		#See if Eve got what she wanted:
		eve_score = 100 * int(self.sync_score(self.Alice, self.Eve))
		if eve_score > 100:
			print("Oops! Nosy Eve synced her machine with Alice's and Bob's !")
		else:
			print("Eve's machine is only " + str(eve_score) + " % " + "synced with Alice's and Bob's and she did " + str(nb_eve_updates) + " updates.") 

#Machine parameters
k = 100
n = 10
l = 10

#Update rule
update_rules = ['hebbian', 'anti_hebbian', 'random_walk']

keyExchange = KeyExchange(k, n, l)
keyExchange.run(update_rules[0])
keyExchange.plot()
