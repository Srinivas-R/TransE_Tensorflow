import tensorflow as tf
import numpy as np

saver = tf.train.import_meta_graph('Output/model.vec.meta')
sess = tf.Session()
saver.restore(sess, 'Output/model.vec')
ents = sess.run('ent_emb:0')
rels = sess.run('rel_emb:0')

with open('Tester/entity2vec.unif','w') as fileout:
	for e in ents:
		temp = [str(x) for x in e.tolist()]
		fileout.write(' '.join(temp) + '\n')

with open('Tester/relation2vec.unif','w') as fileout:
	for r in rels:
		temp = [str(x) for x in r.tolist()]
		fileout.write(' '.join(temp) + '\n')

print('Finished extracting embeddings from saved tensorflow model')
