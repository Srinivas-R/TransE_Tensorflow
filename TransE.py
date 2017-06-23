import tensorflow as tf
import numpy as np

num_ents = 14951
num_rels = 1345
batch_size = 100000
n = 100
L1_flag = 1
margin = 1
nepoch = 1000

def next_batch(data):
	#data is the list of all triples
	idx = np.arange(len(data))
	np.random.shuffle(idx)
	idx = idx[:batch_size]
	pos_triples = data[idx]
	neg_triples = np.array(pos_triples)
	random_ents = np.random.randint(0, num_ents, batch_size)
	prob = np.random.randint(0,1000,batch_size)
	for i in range(len(neg_triples)):
		if prob[i] <= 500:
			neg_triples[i][0] = random_ents[i]
		else:
			neg_triples[i][2] = random_ents[i]

	#return e1, e2, e1_neg, e2_neg, rel
	return pos_triples[:,0], pos_triples[:,2] , neg_triples[:,0], neg_triples[:,2], pos_triples[:,1], 

def read_data(filename):
	train_triples = []
	with open(filename,'r') as filein:
		for line in filein:
			train_triples.append([int(x.strip()) for x in line.split()])
	return train_triples

ent_emb = tf.Variable(tf.random_uniform(shape=[num_ents, n], minval=-1.0, maxval=1.0, dtype=tf.float32),name='ent_emb')
rel_emb = tf.Variable(tf.random_uniform(shape=[num_rels, n], minval=-1.0, maxval=1.0, dtype=tf.float32),name='rel_emb')
global_step = tf.Variable(0, name='global_step', trainable=False)

pos_head = tf.placeholder(tf.int32, [batch_size])
pos_tail = tf.placeholder(tf.int32, [batch_size])
rel = tf.placeholder(tf.int32, [batch_size])
neg_head = tf.placeholder(tf.int32, [batch_size])
neg_tail = tf.placeholder(tf.int32, [batch_size])

pos_head_e = tf.nn.embedding_lookup(ent_emb, pos_head)
pos_tail_e = tf.nn.embedding_lookup(ent_emb, pos_tail)
rel_e = tf.nn.embedding_lookup(rel_emb, rel) 
neg_head_e = tf.nn.embedding_lookup(ent_emb, neg_head)
neg_tail_e = tf.nn.embedding_lookup(ent_emb, neg_tail)

if L1_flag == 1:
	dist_pos = tf.norm(tf.subtract(pos_tail_e, tf.add(pos_head_e, rel_e)) ,axis=1, ord=1)
	dist_neg = tf.norm(tf.subtract(neg_tail_e, tf.add(neg_head_e, rel_e)) ,axis=1, ord=1)
else:
	dist_pos = tf.norm(tf.subtract(pos_tail_e, tf.add(pos_head_e, rel_e)) ,axis=1, ord=2)
	dist_neg = tf.norm(tf.subtract(neg_tail_e, tf.add(neg_head_e, rel_e)) ,axis=1, ord=2)

loss = tf.reduce_sum(tf.maximum(0.0, dist_pos + margin - dist_neg))
tf.summary.scalar('Loss',loss)
merged = tf.summary.merge_all()
saver = tf.train.Saver()

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss,global_step=global_step)
normalizer = tf.nn.l2_normalize(ent_emb, axis=1)
#completed constructing tf graph

data = np.array(read_data('./data/train.txt'))
nbatches = len(data) // batch_size

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(normalizer)
	writer = tf.summary.FileWriter('logs', sess.graph)
	for _ in range(nepoch):
		tot_loss = 0.0
		for __ in range(nbatches):
			ph, pt, nh, nt, r = next_batch(data)
			feed_dict = {pos_head : ph,
						 pos_tail : pt, 
						 rel 	  : r,
						 neg_head : nh,
					 	 neg_tail : nt}

			l,summary, ___ = sess.run([loss,merged,optimizer],feed_dict)
			writer.add_summary(summary, tf.train.global_step(sess, global_step))
			tot_loss += l
		print('Epoch {}\tLoss {}'.format(_,tot_loss))
	saver.save(sess, 'Output/model.vec')
