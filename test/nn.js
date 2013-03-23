var assert = require('assert'),
    nn = require('../lib/nn'),
    util = require('util')

describe('nn', function() {
  it('creates layers properly', function (done) {
    var net = nn({
        layers: [ 5 ]
    })

    assert(net.layers)
    assert(net.layers[1])
    assert.equal(net.layers[1].length, 5)

    var net2 = nn({
        layers: [ 2, 6 ],
        weight: 0.1
    })

    assert(net2.layers)
    assert.equal(net2.layers[1].length, 2)
    assert.equal(net2.layers[2].length, 6)

    done()
  })

  it('trains XOR correctly', function (done) {
    var net = nn({
        weight: 0.1,
        iterations: 100
    })

    net.train([
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] }
    ])

    var output = net.send([1, 0])

    console.log('output', output)

    assert(0.5 < output, 'output was lower than acceptable:' + output)
    assert(1 >= output, 'output was higher than acceptable:' + output)

    done()
  })

  it('trains linear regression correctly', function (done) {
    var net = nn({
        weight: 0.1,
        iterations: 100
    })

    net.train([
        { input: [0.3, 0.2], output: [0.06] },
        { input: [0.2, 0.3], output: [0.06] },
        { input: [0.1, 0.1], output: [0.01] },
    ])

    var output = net.send([0.1, 0.3])

    console.log('output', output)


    done()
  })
})