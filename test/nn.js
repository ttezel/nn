var assert = require('assert'),
    nn = require('../lib/nn'),
    util = require('util')

describe('nn', function() {
    it('creates weights/biases matrices properly', function (done) {
        var net = nn({
            layers: [ 5, 4 ],
            iterations: 2
        })

        assert.equal(net.biases[1].length, 5, 'first hidden layer should have 5 biases (1 for each node): '+util.inspect(net.biases))
        assert.equal(net.biases[2].length, 4, 'second hidden layer should have 4 biases (1 for each node): '+util.inspect(net.biases))
        assert.equal(net.weights[2].length, 4, 'second hidden layer should have 4 weight arrays: '+util.inspect(net.weights))

        net.train({ input: [ 1 ], output: [ 0.3, 0.2 ] })

        assert.equal(net.weights[1].length, 5, 'first hidden layer number of weight arrays should be 5: '+util.inspect(net.weights))
        assert.equal(net.weights[1][0].length, 1, 'weight arrays for the first hidden layer should be of size 1')
        assert.equal(net.weights[1][1].length, 1, 'weight arrays for the first hidden layer should be of size 1')
        assert.equal(net.weights[1][2].length, 1, 'weight arrays for the first hidden layer should be of size 1')
        assert.equal(net.weights[1][3].length, 1, 'weight arrays for the first hidden layer should be of size 1')
        assert.equal(net.weights[1][4].length, 1, 'weight arrays for the first hidden layer should be of size 1')

        // console.log('weights', util.inspect(net.weights))
        // console.log('biases', util.inspect(net.biases))

        done()
    })

    it('trains AND correctly', function (done) {
        var net = nn({
          log: 0
        })

        var trainingData = [
          { input: [ 0, 0 ], output: [ 0 ] },
          { input: [ 0, 1 ], output: [ 0 ] },
          { input: [ 1, 0 ], output: [ 0 ] },
          { input: [ 1, 1 ], output: [ 1 ] }
        ];

        console.log('\nAND test:')
        trainAndTest(net, trainingData)

        done()
    })

    it('trains OR correctly', function (done) {
        var net = nn({
          log: 0
        })

        var trainingData = [
          { input: [ 0, 0 ], output: [ 0 ] },
          { input: [ 0, 1 ], output: [ 1 ] },
          { input: [ 1, 0 ], output: [ 1 ] },
          { input: [ 1, 1 ], output: [ 1 ] }
        ];

        console.log('\nOR test:')
        trainAndTest(net, trainingData)

        done()
    })

    it('interpolates correctly', function (done) {
        var net = nn({
          log: 0
        })

        var trainingData = [
          { input: [ 0 ], output: [ 0 ] },
          { input: [ 0.1 ], output: [ 0.1 ] },
          { input: [ 0.2 ], output: [ 0.2 ] },
          { input: [ 0.3 ], output: [ 0.3 ] },
          { input: [ 0.4 ], output: [ 0.4 ] },

          { input: [ 0.6 ], output: [ 0.6 ] },
          { input: [ 0.7 ], output: [ 0.7 ] },
          { input: [ 0.8 ], output: [ 0.8 ] },
          { input: [ 0.9 ], output: [ 0.9 ] },
          { input: [ 1 ], output: [ 1 ] }
        ];

        net.train(trainingData)

        var stats = net.test({ input: [ 0.5 ], output: [ 0.5 ] })

        var stat = stats.results[0];

        console.log('\n nn interpolation output: %s. desired output: %s. MSE value: %s', stat.output, stat.desiredOutput, stats.mse)

        assert(stats.mse < 0.01)

        done()
    })

    it('trains XOR correctly', function (done) {
        var net = nn({
            log: 0,
            layers: [2]
        })

        var trainingData = [
            { input: [0, 0], output: [0] },
            { input: [0, 1], output: [1] },
            { input: [1, 0], output: [1] },
            { input: [1, 1], output: [0] },
        ];

        console.log('\nXOR test:')
        trainAndTest(net, trainingData)

        done()
    })
})

function trainAndTest (net, trainingData) {
    net.train(trainingData)

    // console.log('net.weights', util.inspect(net.weights, false, 10, true))
    // console.log('net.biases', util.inspect(net.biases, false, 10, true))

    var failure = null

    trainingData.forEach(function (entry) {
        var output = net.send(entry.input)
        console.log(entry.input, 'output:', output)

        var err = Math.abs(entry.output[0] - output)

        if (err > 0.1)
            failure = 'error too large: ' + err
    })

    assert(!failure, failure)
}