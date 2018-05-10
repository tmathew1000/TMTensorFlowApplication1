import tensorflow as tf

#how graph works
graph=tf.get_default_graph()
graph.get_operations()

a=tf.constant(10, name="a")

#operations=graph.get_operations()

b=tf.constant(20,name="b")

#c= a+b (=30)
c=tf.add(a,b,name="c")

#d=a*b = 200
d=tf.multiply(a,b,name="d")
e=tf.multiply(c,d,name="e")

#operations=graph.get_operations()
#print(operations)

sess=tf.Session()
print(sess.run(e))

for op in graph.get_operations(): print(op.name)
