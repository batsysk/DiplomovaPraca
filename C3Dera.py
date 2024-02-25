import cv2
import numpy as np
import tensorflow as tf

with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('./model/c3d_best_model-49.meta')
    saver.restore(sess, tf.train.latest_checkpoint("./model"))

    cap = cv2.VideoCapture('./TestSnippet2/Baseball_001 .mp4')
    input_tensor = sess.graph.get_tensor_by_name('input_x:0')
    output_tensor = sess.graph.get_tensor_by_name('logits:0')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (112, 112))

        input_frames = np.expand_dims(np.array([resized_frame] * 16), axis=0)

        feed_dict = {input_tensor: input_frames, 'training:0': False}
        output_values = sess.run(output_tensor, feed_dict=feed_dict)

    print("Vysledok: ", output_values)
    predicted_class_index = np.argmax(output_values)
    print("Predicted class index:", predicted_class_index)
    
    cap.release()
