const WINDOW_SIZE = 100

document.onload = loadCharts()

function loadCharts() {
    let data = {
        labels: [],
        datasets: [
            {
                label: 'Average time between packets',
                data: [],
                backgroundColor: ['rgba(255, 99, 132, 0.2)'],
                borderColor: ['rgba(255, 99, 132, 1)'],
                tension: 0.4
            },
            {
                label: 'Average entropy',
                data: [],
                backgroundColor: ['rgba(54, 162, 235, 0.2)'],
                borderColor: ['rgba(54, 162, 235, 1)'],
                tension: 0.4
            },
            {
                label: 'Average Hamming distance',
                data: [],
                backgroundColor: ['rgba(255, 206, 86, 0.2)'],
                borderColor: ['rgba(255, 206, 86, 1)'],

                tension: 0.4
            },
        ]
    }

    let config = {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Network metrics'
                }
            },
            interaction: {
                intersect: false
            },
            animation: {
                duration: 50,
                easing: 'easeInOutCubic'
            },
        }
    }

    let chart = new Chart(document.getElementById('chart').getContext('2d'), config)
    let log = document.getElementById('log')
    let alert = document.getElementById('alert')

    io('http://localhost:9001').on('new data', (data) => {
        if (chart.data.labels.length > WINDOW_SIZE) {
            chart.data.labels.shift()
            chart.data.datasets.forEach((dataset) => dataset.data.shift())
        }
        let timestamp = new Date()
        chart.data.labels.push(timestamp.getHours() + ':' + timestamp.getMinutes() + ':' + timestamp.getSeconds())
        for (let i = 0; i < 3; i++)
            chart.data.datasets[i].data.push(data[0][i])
        chart.update()
        
        if (data[1])
            alert.style.opacity = 1
        else
            alert.style.opacity = 0.5

        if (data[2] != null)
            log.innerHTML += data[2] + '\n'
    })
}