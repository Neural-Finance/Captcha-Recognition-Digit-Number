# Captcha_Recognition_Digit
【Framework】This is a captcha recognition algorithm tailored for 4 length digit recognition. It's really straightforward for new beginner.
It's suitable for python3 and tensorflow both gpu or cpu version. (Write with tensorflow 1.3.0)

### Motivation
Once you need to make a solution for some captcha recognition, the first thing you need is the complete datasets (x and label). However, it's almost impossible for you to label a lot of datasets. Thus, here is a solution for this case. Because the captcha may made up by different fonts of digit or english character. You can use a package called 'Captcha' to easily generate a lot of training data (with label). You can give it a random font, figure size, character position, noise background, rotate angle, etc...Once you have this training data which share the samilar distribution with the test data. Then, you can send this data into a CNN network to do supervised machine learning. In this project, it can be divided into 4 steps. 

--------------------------------------

**Main.py** 

1.check the environment and make folder; 2.generate the training data; 3.save the training data as tf recoder; 4.training the network; 5.test the captcha

--------------------------------------

**Alexnet.py**

a poverful CNN network which can help you extract features (information) from the picture

![Image text](https://github.com/Neural-Finance/Captcha-Recognition-Digit-Number/blob/master/1.png)

--------------------------------------

### Input and Output
Input is the image, which is jpg format, we will read in it, treat it as a array and then save it by using tfrecord.

Output is a classification possibility, for example [0,0,0,0,0,0,0,0,0.2,0.8], because 10 has the highest possibility, thus we think this label is 10.

![Image text](https://github.com/Neural-Finance/Captcha-Recognition-Digit-Number/blob/master/3_3.png)

### Performance
After about 5000 iterations, we got 100% accuracy for this task:

![Image text](https://github.com/Neural-Finance/Captcha-Recognition-Digit-Number/blob/master/2.png)

The prediction result:

![Image text](https://github.com/Neural-Finance/Captcha-Recognition-Digit-Number/blob/master/4.png)


### Further Research
Beacuse this is a simple case, sometimes, the captcha can be a n-length picture. Thus, we have to cut the picture and train each number individually. If you have interested in this more sophisticated case, please refer to my corresponding work in this field. However, if you sure that the captach is fixed length, you can just leverage this porject.

**Thanks for your patience to read here. I will be really glad, if you like this work.**
