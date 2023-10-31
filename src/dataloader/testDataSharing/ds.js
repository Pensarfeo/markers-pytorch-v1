const fs              = require('fs');
const { spawn, fork } = require('child_process');

const path_a = 'pipe_c';
const path_b = 'pipe_b';
let fifo_b   = spawn('mkfifo', [path_b]);  // Create Pipe B

console.log('creating array')
const arrr = new Array(256*256*3*64).fill(1)

const arr= new Uint8Array(256*256*3*64)
console.log('arr:', arr.length*8)
console.log('pages:', arr.length*8/65536) 

arrr.forEach((e, i) => {
  arrr[i] = 100
})

console.log('- created array')


fifo_b.on('exit', function(status) {
    console.log('Created Pipe B');

    const fd   = fs.openSync(path_b, 'r+');
    let fifoRs = fs.createReadStream(null, { fd });
    let fifoWs = fs.createWriteStream(path_a);

    console.log('Ready to write')

    setTimeout(() => {
        console.log('-----   Send packet   -----');
        fifoWs.write(arr);
        console.log('-----   Packet sent   -----');
    }, 1000);  // Write data at 1 second interval

    fifoRs.on('data', data => {

        // const now_time  = new Date();
        // const sent_time = new Date(data.toString());
        // const latency   = (now_time - sent_time);

        console.log('----- Received packet -----');
        // console.log(data);
        // console.log('    Latency: ' + latency.toString() + ' ms');
    });
});