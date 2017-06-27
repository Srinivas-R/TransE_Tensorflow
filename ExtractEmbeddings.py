import tensorflow as tf
import numpy as np

saver = tf.train.import_meta_graph('logs/model.vec.meta')
sess = tf.Session()
saver.restore(sess, 'logs/model.vec')
ents = sess.run('ent_emb:0')
rels = sess.run('rel_emb:0')
ents_class = sess.run('ent_class:0')
rel_head_class = sess.run('rel_head_class:0')
rel_tail_class = sess.run('rel_tail_class:0')

with open('Tester/entity2vec.unif','w') as fileout:
	for e in ents:
		temp = [str(x) for x in e.tolist()]
		fileout.write(' '.join(temp) + '\n')

with open('Tester/relation2vec.unif','w') as fileout:
	for r in rels:
		temp = [str(x) for x in r.tolist()]
		fileout.write(' '.join(temp) + '\n')

with open('Tester/ent_class.unif','w') as fileout:
	for e in ents_class:
		temp = [str(x) for x in e.tolist()]
		fileout.write(' '.join(temp) + '\n')

with open('Tester/rel_head_class.unif','w') as fileout:
	for r in rel_head_class:
		temp = [str(x) for x in r.tolist()]
		fileout.write(' '.join(temp) + '\n')

with open('Tester/rel_tail_class.unif','w') as fileout:
	for r in rel_tail_class:
		temp = [str(x) for x in r.tolist()]
		fileout.write(' '.join(temp) + '\n')

print('Finished extracting embeddings from saved tensorflow model')
