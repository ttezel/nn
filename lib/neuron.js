var util = require('util'),
    activation = require('./activation')

module.exports = Neuron

/*
    Neuron class - Adaptive Linear Neuron (adaline)

    weights are applied to input and used to determine the output
 */
function Neuron (opts) {
    this.weights = [];

    // number of incoming connections (weights to set)
    if (!opts.n)
        opts.n = 3
    if (!opts.weight)
        opts.weight = 0.1
    if (!opts.bias)
        opts.bias = -0.2
    if (!opts.activation)
        opts.activation = 'hyperbolic'

    if (typeof opts.weight === 'number') {
        for (var i = 0; i < opts.n; i++) {
            this.weights.push(opts.weight)
        }
    } else if (Array.isArray(opts.weight)) { 
        this.weights = opts.weight.slice()
    } else {
        throw new Error('opts.weight must be Number or Array. Got: '+util.inspect(opts.weight, true, 10, true))
    }

    this.opts = opts

    this.bias = opts.bias
}

/**
 * Send the Neuron an input array, get output back.
 * 
 * @param  {Array} input    input array of numbers
 * @return {Number}         output
 */
Neuron.prototype.send = function (input) {
    var self = this
    var weightedSum = 0

    // compute weighted sum of the input array
    input.forEach(function (value, index) {
        var weight = self.weights[index] || 0
        weightedSum += (value * weight)
    })

    var activationFn = activation[this.opts.activation]

    if (!activationFn)
        throw new Error('this.activate must be a supported activation function name. Got: '+this.activate)

    weightedSum = weightedSum - this .bias

    // run the weighted sum thru the activation function
    var output = activationFn.activate(weightedSum)

    return output
}
