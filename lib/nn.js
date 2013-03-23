var util = require('util'),
    Neuron = require('./neuron')

module.exports = function (opts) {
    return new NN(opts)
}

/**
 * NN class (implements a feedforward neural network)
 * 
 * @param {Object} opts
 */
function NN (opts) {
    var self = this

    if (!opts)
        opts = {}
    // normalize opts and choose sane defaults
    if (!opts.layers)
        opts.layers = [ 3 ]
    if (!opts.weight)
        opts.weight = 0.1
    if (!opts.bias)
        opts.bias = -0.2
    if (!opts.iterations)
        opts.iterations = 200
    if (!opts.maxError)
        opts.maxError = 0.005
    if (!opts.activation)
        opts.activation = 'logistic'
    if (!opts.learningRate)
        opts.learningRate = 0.3
    if (!Array.isArray(opts.layers))
        throw new Error('nn: opts.layers must be an array. Got: '+opts.layers)

    this.opts = opts

    // this is a 2D matrix representing our layers of neurons
    this.layers =  [];

    var optLayers = this.opts.layers

    // initialize each hidden layer - leave first layer
    // empty to use as input layer
    for (var i = 0; i < optLayers.length; i++) {
        this.layers[i + 1] = [];

        // initialize each neuron in the hidden layer
        for (var j = 0; j < optLayers[i]; j++) {
            this.layers[i + 1][j] = new Neuron({
                n: optLayers[i],
                weight: this.opts.weight,
                bias: this.opts.bias
            })
        }
    }
}

NN.prototype.train = function (trainingData) {
    if (Array.isArray(trainingData))
        return trainingData.forEach(this.train.bind(this))

    var self = this
    var input = trainingData.input
    var desiredOutput = trainingData.output

    // console.log('this.layers', util.inspect(this.layers, true, 10, true))

    // add output layer
    if (!this.outputLayer) {
        var outputLayerIndex = this.layers.push([]) - 1
        this.outputLayer = this.layers[outputLayerIndex];
        var numIncoming = this.layers[outputLayerIndex-1].length

        // initialize output layer
        for (var i = 0; i < desiredOutput.length; i++) {
            this.layers[outputLayerIndex][i] = new Neuron({
                n: numIncoming,
                weight: this.opts.weight,
                bias: this.opts.bias
            })
        }
    }

    // console.log('this.layers str', this.layers)

    var iterations = this.opts.iterations

    while (iterations--) {
        // propagate `input` forward thru the layers
        // and populate this.outputs
        this.send(input)

        // back-propagate error values thru layers
        // and correct weights/delta values
        this.backPropagate(desiredOutput)
    }
}

/**
 * Propagate `input` thru the neural net, and 
 * populate this.outputs
 * 
 * @param  {Array or Object} input
 */
NN.prototype.send = function (input) {
    // initialize input layer
    this.layers[0] = input

    // keep track of the outputs of each neuron
    this.outputs = [];

    // set output of first (input) layer
    this.outputs[0] = input

    // now propagate forward thru the hidden layers
    for (var layer = 1; layer < this.layers.length; layer++) {
        // initialize output array for this layer
        this.outputs[layer] = [];

        // send each neuron input
        for (var n = 0; n < this.layers[layer].length; n++) {
            var neuron = this.layers[layer][n]
            var prevLayerOutput = this.outputs[layer - 1]
            var response = neuron.send(prevLayerOutput)
            this.outputs[layer][n] = response
        }
    }

    // return output of last layer
    return this.outputs[this.layers.length - 1];
}

NN.prototype.backPropagate = function (desiredOutput) {
    this.errorSigs = [];
    this.weightDeltas = [];

    var outputLayerIndex = this.outputs.length - 1
    var outputLayer = this.outputs[outputLayerIndex];

    this.errorSigs[outputLayerIndex] = [];

    // console.log('this.outputs', this.outputs)

    // determine weight deltas for output layer
    for (var n = 0; n < outputLayer.length; n++) {
        var neuron = this.layers[outputLayerIndex][n]

        var desiredOut = desiredOutput[n] || 0;
        var neuronOut = outputLayer[n] || 0;

        var neuronError = desiredOut - neuronOut
        var errorSig = neuronOut * (1 - neuronOut) * neuronError
            
        var weightDelta = this.opts.learningRate * errorSig * neuronOut
    
        var prevLayerOutput = this.outputs[outputLayerIndex - 1]

        this.errorSigs[outputLayerIndex][n] = errorSig

        // update neuron connection weights
        for (var p = 0; p < prevLayerOutput.length; p++) {
            neuron.weights[p] += prevLayerOutput[p] * weightDelta
        }

        // update neuron bias
        var biasDelta = this.opts.learningRate * errorSig

        neuron.bias += biasDelta
    }

    var lastHiddenLayerNum = outputLayerIndex - 1

    // iterate backwords thru the rest of the hidden layers
    for (var layer = lastHiddenLayerNum; layer > 0; layer--) {
        var nextLayer = this.layers[layer + 1];
        var nextLayerErrorSigs = this.errorSigs[layer + 1];

        this.errorSigs[layer] = [];

        // determine error of each neuron's output
        for (var n = 0; n < this.outputs[layer].length; n++) {
            var neuronOut = this.outputs[layer][n] || 0;

            var neuronError = 0

            // determine errors for each connection to this neuron
            for (var d = 0; d < nextLayer.length; d++) {
                var nextLayerNeuron = nextLayer[d]
                neuronError += nextLayerErrorSigs[d] * nextLayerNeuron.weights[n]
            }
            
            var errorSig = neuronOut * (1 - neuronOut) * neuronError
            this.errorSigs[layer][n] = errorSig

            var weightDelta = this.opts.learningRate * errorSig * neuronOut

            var prevLayerOutput = this.outputs[outputLayerIndex - 1]

            // update neuron connection weights
            for (var p = 0; p < prevLayerOutput.length; p++) {
                neuron.weights[p] += prevLayerOutput[p] * weightDelta
            }

            // update neuron bias
            var biasDelta = this.opts.learningRate * errorSig

            neuron.bias += biasDelta
        }
    }
}

