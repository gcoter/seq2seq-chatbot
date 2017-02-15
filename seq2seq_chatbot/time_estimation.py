""" Classes and functions to estimate remaining time for different processes.
"""
def seconds2minutes(time):
	""" Helper to display time. """
	hours = int(time) // 3600
	minutes = (int(time) % 3600) // 60
	seconds = (int(time) % 3600) % 60
	return hours, minutes, seconds

class ReadingTimeEstimator(object):
	""" Estimate the remaining time before the reading process ends. """
	def __init__(self,num_conversations):
		self.num_conversations = num_conversations
		self.previous_times = []

	def estimate_time_spent_per_conversation(self):
		return sum(self.previous_times)/len(self.previous_times)

	def get_remaining_time(self,
	                       num_conversations_left,
						   num_conversations_read_in_time_spent,
						   time_spent):
		if num_conversations_read_in_time_spent > 0:
			self.previous_times.append(
				float(time_spent)/num_conversations_read_in_time_spent)
			remaining_time = self.estimate_time_spent_per_conversation() * num_conversations_left
			return remaining_time
		return None

class TrainingTimeEstimator(object):
	"""
	"""
	def __init__(self,num_steps_per_epoch):
		self.num_steps_per_epoch = num_steps_per_epoch
		self.previous_times = []

	"""
	"""
	def estimate_time_spent_per_step(self):
		return sum(self.previous_times)/len(self.previous_times)

	"""
	"""
	def get_remaining_time(self,
						   num_epochs_left,
						   num_steps_left_in_current_epoch,
						   num_steps_processed_in_time_spent,
						   time_spent):
		if num_steps_processed_in_time_spent > 0:
			self.previous_times.append(float(time_spent)/num_steps_processed_in_time_spent)
		current_epoch_remaining_time = self.estimate_time_spent_per_step() * num_steps_left_in_current_epoch
		remaining_time_per_epoch = self.estimate_time_spent_per_step() * self.num_steps_per_epoch
		return current_epoch_remaining_time + num_epochs_left * remaining_time_per_epoch
