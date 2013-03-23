#`nn`

#Simple, easy to use Neural Network for node.js

##Install
```
    npm install nn
```

##Usage
```javascript
var nn = require('nn')

var net = nn()

// or with input/output arrays
net.train({ input: [1, 2, 3], output: [2, 4, 6] })

// or train in bulk
net.train([
    { input: [0.13, 1.4, 0.6], output: [4.1, 1.2, 6.8] },
    { input: [0.8, 0.6, 4.4], output: [0.9, 12, 5.4] },
    ...
])

// send it new input and see its output
var output = net.send([0.1, 0.2])
```

