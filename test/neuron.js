var assert = require('assert'),
    util = require('util'),
    Neuron = require('../lib/neuron')

describe('Neuron', function () {
    it('outputs correctly with hyperbolic activation', function (done) {
        var opts = {
            layers: 1,
            n: 10,
            activate: 'hyperbolic'       
         }

         var n = new Neuron(opts)

         // - generate 100 random input arrays
         // - send them to the neuron
         // - make sure the output is in hyperbolic range
         //   for each input
         for (var i = 0; i < 100; i++) {
            var input = generateInput()
            var output = n.send(input)

            // make sure output is in hyperbolic range (-1,1)
            checkRange(output, -1, 1)
         }

         done()
    })

    it('outputs correctly with logistic activation', function (done) {
        var opts = {
            layers: 1,
            n: 10,
            activate: 'logistic'
        }

        var n = new Neuron(opts)

        // - generate 100 random input arrays
        // - send them to the neuron
        // - make sure the output is in logistic range
        //   for each input
        for (var i = 0; i < 100; i++) {
           var input = generateInput()
           var output = n.send(input)

           // make sure output is in logistic range (0,1)
           checkRange(output, 0, 1)
        }

        done()
    })
})

// generate a randomly-sized array of random numbers
function generateInput () {
    var maxSize = 50
    var maxNum = 10

    var inputSize = Math.floor(Math.random() * maxSize) + 1
    var input = [];

    // push a bunch of random numbers
    for (var i = 0; i < inputSize; i++) {
        input.push(Math.random() * maxNum)
    }

    return input
}

function print (thing) {
    return util.inspect(thing, true, 10, true)
}

function checkRange (value, min, max) {
    assert(typeof value === 'number', print(value))
    assert(value >= min, print(value))
    assert(value <= max, print(value))
}
