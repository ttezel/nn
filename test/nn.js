var assert = require('assert'),
    nn = require('../lib/nn'),
    util = require('util')

describe('simple-nn', function() {

    it('initializes weights/biases matrices properly', function (done) {
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

    it('trains linear correctly: y = x', function (done) {
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

    it('trains linear interpolation correctly: y = 0.7x + 0.11', function (done) {
        var net = nn({
            log: 0,
            errorThresh: 0.00001,
            momentum: 0.8
        })

        var trainingData = [
            { input: [ 0 ], output: [ 0.11 ] },
            { input: [ 0.1 ], output: [ 0.18 ] },
            { input: [ 0.2 ], output: [ 0.25 ] },
            { input: [ 0.3 ], output: [ 0.32 ] },
            { input: [ 0.4 ], output: [ 0.39 ] }
        ];

        net.train(trainingData)

        var stats = net.test([
            { input: [ 0.25 ], output: [ 0.285 ] }
        ])

        var stat = stats.results[0];

        console.log('\n nn regression output: %s. desired output: %s. MSE value: %s', stat.output, stat.desiredOutput, stats.mse)

        assert(stats.mse < 0.001)

        done()
    })

    it('trains regression correctly: y = sin(x)', function (done) {
        var net = nn()

        // this example shows how we could train it to approximate sin(x)
        // from a random set of input/output sets.
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

        // send it an input array to see its trained output
        var output = net.send([ 0.5 ]) // => 0.48031129953896595

        console.log('trained sin output for x = 0.5: %s. desiredOutput: %s', output, Math.sin(0.5))

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