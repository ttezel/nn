# `nn`

# Fast and simple Neural Network for node.js

`nn` is a Neural Network library built for performance and ease of use. It is easy to configure and has sane defaults. You can use it for tasks such as pattern recognition and function approximation. 

# Install
```
npm install nn
```

# Usage
```javascript
var nn = require('nn')

var net = nn()

// this example shows how we could train it to approximate sin(x)
// from a random set of input/output data.
net.train([
    { input: [ 0.5248588903807104 ],    output: [ 0.5010908941521808 ] },
    { input: [ 0 ],                     output: [ 0 ] },            
    { input: [ 0.03929789311951026 ],   output: [ 0.03928777911794752 ] },
    { input: [ 0.07391509227454662 ],   output: [ 0.07384780553540908 ] },
    { input: [ 0.11062344848178328 ],   output: [ 0.1103979598825075 ] },
    { input: [ 0.14104655454866588 ],   output: [ 0.14057935309092454 ] },
    { input: [ 0.06176552915712819 ],   output: [ 0.06172626426511784 ] },
    { input: [ 0.23915000406559558 ],   output: [ 0.2368769073277496 ] },
    { input: [ 0.27090200221864513 ],   output: [ 0.267600651550329 ] },
    { input: [ 0.15760037200525404 ],   output: [ 0.1569487719674096 ] },
    { input: [ 0.19391102618537845 ],   output: [ 0.19269808506017222 ] },
    { input: [ 0.42272064974531537 ],   output: [ 0.4102431360805792 ] },
    { input: [ 0.5248469677288086 ],    output: [ 0.5010805763172892 ] },
    { input: [ 0.4685300185577944 ],    output: [ 0.45157520770441445 ] },
    { input: [ 0.6920387226855382 ],    output: [ 0.6381082150316612 ] },
    { input: [ 0.40666140150278807 ],   output: [ 0.3955452139761714 ] },
    { input: [ 0.011600911058485508 ],  output: [ 0.011600650849602313 ] },
    { input: [ 0.404806485096924 ],     output: [ 0.39384089298297537 ] },
    { input: [ 0.13447276877705008 ],   output: [ 0.13406785820465852 ] },
    { input: [ 0.22471809106646107 ],   output: [ 0.222831550102815 ] } 
])

// send it a new input to see its trained output
var output = net.send([ 0.5 ]) // => 0.48031129953896595
```

# methods

## `var net = nn(opts)`

Creates a Neural Network instance. Pass in an optional `opts` object to configure the instance. Any values specified in `opts` will override the corresponding defaults.

The default configuration is shown below:
```javascript
{
    // hidden layers eg. [ 4, 3 ] => 2 hidden layers, with 4 neurons in the first, and 3 in the second.
    layers: [ 3 ],
    // maximum training epochs to perform on the training data
    iterations: 20000,
    // maximum acceptable error threshold
    errorThresh: 0.0005,
    // activation function ('logistic' and 'hyperbolic' supported)
    activation: 'logistic',
    // learning rate
    learningRate: 0.4,
    // learning momentum
    momentum: 0.5,
    // logging frequency to show training progress. 0 = never, 10 = every 10 iterations.
    log: 0   
}
```

## `net.train(trainingData)`

Train your neural network instance, using `trainingData`. You can pass in a single training entry as an object with `input` and `output` keys, or an array of training entries. The network will train itself from the supplied training data, until the error threshold has been reached, or the max number of iterations has been reached.

## `net.send(input)`

Sends your neural network the input data and returns its output. `input` is an array of numbers. Typically you'll call this function after training your network.

## `net.test(testData)`

Runs your neural network against `testData` and returns an object with statistics about the performance of the network against the test data. `testData` can be a single object with `input` and `output` keys, or an array of those objects. Typically you'll call this function after training your network.

## `net.toJson()`

Returns a JSON string representing the state of the neural network. You can later use `nn.fromJson()` to get back the neural network from the JSON string.

## `nn.fromJson(jsonString)`

Load a neural network instance from the JSON representation. Pass in `jsonString` as a string.


-------

# License 

(The MIT License)

Copyright (c) by Tolga Tezel <tolgatezel11@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

