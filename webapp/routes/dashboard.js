const express = require('express')
const router = express.Router();
const app = express();
const http = require('http');
const server = http.createServer(app);
const { Server } = require('socket.io');
const io = new Server(server, {
  cors: {
    origin: "http://localhost:9000"
  }
});

let socket

io.on('connection', (s) => {
  socket = s
  console.log('Connected to front-end')
})
server.listen(9001)

router.post('/', (req, res) => {
  socket.emit('new data', req.body)
  res.status(200).end()
})

router.get('/', (req, res) => {
  res.render('dashboard')
})

module.exports = router;