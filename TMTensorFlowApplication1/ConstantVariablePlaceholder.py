import tensorflow as tf

#this is a variable
zero = tf.Variable(0)

#this is a constant
one= tf.constant(1)

#Add Variable and Constant and assign it to a new value. new_value = zero + one
new_value=tf.add(zero, one)

#Variable value can be changed. constants cannot be changed. Zero variable has a new value
update=tf.assign(zero, new_value)

# Initialize all variables
init_op = tf.global_variables_initializer()

sess = tf.Session()

#whenever you have variables in the code, you have to initialize, this is standard
sess.run (init_op)

#print (sess.run(zero))
#print (sess.run(new_value))

for _ in range(5):
        sess.run(update)
        print(sess.run(zero))

#string Operations
hello=tf.constant("hello")
world=tf.constant("world")
helloworld = tf.add(hello, world)
print (sess.run(helloworld))

#placeholder
a=tf.placeholder(tf.float32)
b=a*2

#at some point when a has a value, b = twice a

#feeding a placeholder with scalar
result = sess.run(b,feed_dict={a:3}) #use feed_dict as it is moe readable
#result=sess.run (b,{a:3})
print(result)



#scalar is for rank 0, now let us fee a placeholder with a vector of rank1
#feeding a placeholder with vector of rank 1
result = sess.run(b, feed_dict={a:[3,4,5]})
print(result)

#feeding a placeholder with multidimensional vector 3X4X2
dictionary= {a:[[[1,2,3], [4,5,6], [7,8,9], [10,11,12]],[[13, 14,15], [16,17,18], [19,20,21],[22,23,24]]]}
result=sess.run(b,feed_dict=dictionary)
print(result)

sess.close()


#Another way of using the session - Popular way
with tf.Session() as sess:
        result = sess.run(hello+world)
        print(result)
        #many more lines
        #all lines will be executed within this sesion
        #no need of explicitly closing the session

#there is 3rd way, but it is for notebook, not needed here.



