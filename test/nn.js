var assert = require('assert'),
    nn = require('../lib/nn'),
    util = require('util')

describe('nn', function() {
    it('creates weights/biases matrices properly', function (done) {
        var net = nn({
            layers: [ 5, 4 ],
            log: 1,
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
          layers: [3],
          log: 100
        })

        net.train([
          { input: [ 0, 0 ], output: [ 0 ] },
          { input: [ 0, 1 ], output: [ 0 ] },
          { input: [ 1, 0 ], output: [ 0 ] },
          { input: [ 1, 1 ], output: [ 1 ] }
        ])

        var output = net.send([ 0, 1 ])
        var target = 0

        assert(Math.abs(output - target) < 0.1)

        done()
    })

    it.skip('trains XOR correctly', function (done) {
        var net = nn({
            log: 100,
            layers: [3,3],
            iterations: 5000
        })

        net.train([
            { input: [0, 0], output: [0] },
            { input: [0, 1], output: [1] },
            { input: [1, 0], output: [1] },
            { input: [1, 1], output: [0] }
        ])

        var output = net.send([1, 0])

        console.log('trained output', output)

        done()
    })
})