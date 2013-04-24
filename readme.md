#`nn`

#Fast and simple Neural Network for node.js

#Install
```
npm install nn
```

#Usage
```javascript
var nn = require('nn')

var net = nn()

// train the neural network with input/output sets
net.train({ input: [1, 2, 3], output: [0, 1, 0] })

// or train in bulk with an array of input/output sets
net.train([
    { input: [0.1, 0.4, 0.6], output: [0.18, 0.2, 0.82] },
    { input: [0.8, 0.6, 0.4], output: [0.9, 0.12, 0.054] },
    ...
])

// send it an input array to see its trained output
var output = net.send([0.1, 0.2])
```

#methods

##`var net = nn(opts)`

Creates a Neural Network instance. Pass in an optional `opts` object to configure the instance. Any values specified in `opts` will override the corresponding defaults.

The default configuration is shown below:
```javascript
{
    // hidden layers eg. [ 4, 3 ] => 2 hidden layers, with 4 neurons in the first, and 3 in the second.
    layers: [ 3 ],
    // training epochs to perform on the training data
    iterations: 20000,
    // minimum acceptable error threshold
    errorThresh: 0.005,
    // activation function ('logistic' and 'hyperbolic' supported)
    activation: 'logistic',
    // learning rate
    learningRate: 0.3,
    // learning momentum
    momentum: 0.1,
    // initial bias value for each neuron
    bias: 0.1,
    // logging frequency to show training progress. 0 = never, 10 = every 10 iterations.
    log: 0   
}
```

##`net.train(trainingData)`

Train your `nn` instance, using `trainingData`. You can pass in a single training entry as an object with `input` and `output` keys, or an array of training entries. By default, `nn` will perform 2000 epochs of training on the data passed in, or less if it manages to achieve an error margin of less than `errorThresh` on the data.

##`net.send(input)`

Send your `nn` instance input data to see its output. Typically you'll want to call this function after training your instance.

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

