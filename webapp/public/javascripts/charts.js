const WINDOW_SIZE = 200
const NUM_CHARTS = 3

google.charts.load('current', { 'packages': ['corechart'] });
google.charts.setOnLoadCallback(drawCharts);

function drawCharts() {
    let charts = []
    let datatables = []

    for (let i = 0; i < NUM_CHARTS; i++) {
        let datatable = new google.visualization.DataTable()
        datatable.addColumn('number')
        datatable.addColumn('number')
        datatables.push(datatable)
        charts.push(new google.visualization.LineChart(document.getElementById('chart_div' + (i + 1))))
    }

    charts[0].draw(datatables[0], { title: 'Average time between packets', curveType: 'function', legend: 'none' })
    charts[1].draw(datatables[1], { title: 'Average entropy', curveType: 'function', legend: 'none' })
    charts[2].draw(datatables[2], { title: 'Average Hamming distance', curveType: 'function', legend: 'none' })

    io('http://localhost:9001').on('new data', (data) => {
        if (datatables[0].getNumberOfRows() > WINDOW_SIZE) {
            for (dt of datatables) dt.removeRow(0)
            for (dt of datatables)
                for (let r = 0; r < WINDOW_SIZE; r++)
                    dt.setValue(r, 0, dt.getValue(r, 0) - 1)
        }
        for (let i = 0; i < NUM_CHARTS; i++)
            datatables[i].addRow([datatables[i].getNumberOfRows(), data[0][i]])


        charts[0].draw(datatables[0], { title: 'Average time between packets', curveType: 'function', legend: 'none' })
        charts[1].draw(datatables[1], { title: 'Average entropy', curveType: 'function', legend: 'none' })
        charts[2].draw(datatables[2], { title: 'Average Hamming distance', curveType: 'function', legend: 'none' })
        if (data[1]) document.getElementById('alert').style.opacity = 1
        else document.getElementById('alert').style.opacity = 0.5
    })
}