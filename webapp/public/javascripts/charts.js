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

    let chart = new Chart(document.getElementById('chart1').getContext('2d'), config)

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
    })
}