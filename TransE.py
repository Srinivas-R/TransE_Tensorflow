import tensorflow as tf
import numpy as np

num_ents = 14951
num_rels = 1345
batch_size = 50000
n = 100
#embedding size for class vectors
n2 = 100
L1_flag = 1
margin = 1
nepoch = 3000

def next_batch(data):
	#data is the list of all triples
	idx = np.arange(len(data))
	np.random.shuffle(idx)
	idx = idx[:batch_size]
	pos_triples = data[idx]
	neg_triples = np.array(pos_triples)
	random_ents = np.random.randint(0, num_ents, batch_size)
	random_rels = np.random.randint(0, num_rels, batch_size)
	prob = np.random.randint(0,1000,batch_size)
	for i in range(len(neg_triples)):
		if prob[i] <= 500:
			neg_triples[i][0] = random_ents[i]
		else:
			neg_triples[i][2] = random_ents[i]

	#return e1, e2, e1_neg, e2_neg, rel, rel_neg
	return pos_triples[:,0], pos_triples[:,2] , neg_triples[:,0], neg_triples[:,2], pos_triples[:,1], random_rels

def read_data(filename):
	train_triples = []
	with open(filename,'r') as filein:
		for line in filein:
			train_triples.append([int(x.strip()) for x in line.split()])
	return train_triples

ent_emb = tf.Variable(tf.random_uniform(shape=[num_ents, n], minval=-1.0, maxval=1.0, dtype=tf.float32),name='ent_emb')
rel_emb = tf.Variable(tf.random_uniform(shape=[num_rels, n], minval=-1.0, maxval=1.0, dtype=tf.float32),name='rel_emb')

ent_class = tf.Variable(tf.random_uniform(shape=[num_ents,n2], minval=-1.0, maxval=1.0, dtype=tf.float32),name='ent_class')
rel_head_class = tf.Variable(tf.random_uniform(shape=[num_rels,n2], minval=-1.0, maxval=1.0, dtype=tf.float32),name='rel_head_class')
rel_tail_class= tf.Variable(tf.random_uniform(shape=[num_rels,n2], minval=-1.0, maxval=1.0, dtype=tf.float32),name='rel_tail_class')

global_step1 = tf.Variable(0, name='global_step1', trainable=False)
global_step2 = tf.Variable(0, name='global_step2', trainable=False)
global_step3 = tf.Variable(0, name='global_step3', trainable=False)

pos_head = tf.placeholder(tf.int32, [batch_size])
pos_tail = tf.placeholder(tf.int32, [batch_size])
rel = tf.placeholder(tf.int32, [batch_size])
neg_head = tf.placeholder(tf.int32, [batch_size])
neg_tail = tf.placeholder(tf.int32, [batch_size])
rel_neg = tf.placeholder(tf.int32, [batch_size])

pos_head_e = tf.nn.embedding_lookup(ent_emb, pos_head)
pos_tail_e = tf.nn.embedding_lookup(ent_emb, pos_tail)
rel_e = tf.nn.embedding_lookup(rel_emb, rel) 
neg_head_e = tf.nn.embedding_lookup(ent_emb, neg_head)
neg_tail_e = tf.nn.embedding_lookup(ent_emb, neg_tail)

head_class_e = tf.nn.embedding_lookup(ent_class, pos_head)
tail_class_e = tf.nn.embedding_lookup(ent_class, pos_tail)
r_h_class_e = tf.nn.embedding_lookup(rel_head_class, rel) 
r_t_class_e = tf.nn.embedding_lookup(rel_tail_class, rel)
r_h_neg_e = tf.nn.embedding_lookup(rel_head_class, rel_neg)
r_t_neg_e = tf.nn.embedding_lookup(rel_tail_class, rel_neg)


if L1_flag == 1:
	dist_pos = tf.norm(tf.subtract(pos_tail_e, tf.add(pos_head_e, rel_e)) ,axis=1, ord=1)
	dist_neg = tf.norm(tf.subtract(neg_tail_e, tf.add(neg_head_e, rel_e)) ,axis=1, ord=1)
	
	dist_head_class_pos = tf.norm(tf.subtract(head_class_e, r_h_class_e), axis=1, ord=1)
	dist_head_class_neg = tf.norm(tf.subtract(head_class_e, r_h_neg_e), axis=1, ord=1)

	dist_tail_class_pos = tf.norm(tf.subtract(tail_class_e, r_t_class_e), axis=1, ord=1)
	dist_tail_class_neg = tf.norm(tf.subtract(tail_class_e, r_t_neg_e), axis=1, ord=1)
else:
	dist_pos = tf.norm(tf.subtract(pos_tail_e, tf.add(pos_head_e, rel_e)) ,axis=1, ord=2)
	dist_neg = tf.norm(tf.subtract(neg_tail_e, tf.add(neg_head_e, rel_e)) ,axis=1, ord=2)
	
	dist_head_class_pos = tf.norm(tf.subtract(head_class_e, r_h_class_e), axis=1, ord=2)
	dist_head_class_neg = tf.norm(tf.subtract(tail_class_e, r_t_class_e), axis=1, ord=2)

	dist_tail_class_pos = tf.norm(tf.subtract(tail_class_e, r_t_class_e), axis=1, ord=2)
	dist_tail_class_neg = tf.norm(tf.subtract(tail_class_e, r_t_neg_e), axis=1, ord=2)

loss = tf.reduce_sum(tf.maximum(0.0, dist_pos + margin - dist_neg))
loss_head = tf.reduce_sum(tf.maximum(0.0, dist_head_class_pos + margin - dist_head_class_neg))
loss_tail = tf.reduce_sum(tf.maximum(0.0, dist_tail_class_pos + margin - dist_tail_class_neg))

tf.summary.scalar('Loss_TransE',loss)
tf.summary.scalar('Loss_HeadClass',loss_head)
tf.summary.scalar('Loss_TailClass',loss_tail)

merged = tf.summary.merge_all()
saver = tf.train.Saver()

optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss,global_step=global_step1)
optimizer2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_head,global_step=global_step2)
optimizer3 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_tail,global_step=global_step3)

ent_normalizer = tf.assign(ent_emb, tf.nn.l2_normalize(ent_emb, dim=1))
rel_normalizer = tf.assign(rel_emb, tf.nn.l2_normalize(rel_emb, dim=1))
class_ent_normalizer = tf.assign(ent_class, tf.nn.l2_normalize(ent_class, dim=1))
class_rel_head_normalizer = tf.assign(rel_head_class, tf.nn.l2_normalize(rel_head_class, dim=1))
class_rel_tail_normalizer = tf.assign(rel_tail_class, tf.nn.l2_normalize(rel_tail_class, dim=1))

#completed constructing tf graph

data = np.array(read_data('./data/train.txt'))
nbatches = len(data) // batch_size

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(ent_normalizer)
	sess.run(rel_normalizer)
	sess.run(class_ent_normalizer)
	sess.run(class_rel_head_normalizer)
	sess.run(class_rel_tail_normalizer)
	writer = tf.summary.FileWriter('logs', sess.graph)
	for _ in range(nepoch):
		epoch_loss = 0.0
		for __ in range(nbatches):
			ph, pt, nh, nt, r, r_neg = next_batch(data)
			feed_dict = {pos_head : ph,
						 pos_tail : pt, 
						 rel 	  : r,
						 neg_head : nh,
					 	 neg_tail : nt,
					 	 rel_neg  : r_neg}

			l,summary, a,b,c = sess.run([loss,merged,optimizer1,optimizer2,optimizer3],feed_dict)
			writer.add_summary(summary, tf.train.global_step(sess, global_step1))
			
			epoch_loss += l
		print('Epoch {}\tLoss {}'.format(_,epoch_loss))
	saver.save(sess, 'logs/model.vec')
