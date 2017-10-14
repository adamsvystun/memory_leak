    // var ctx1 = document.getElementById("year-chart");
    // console.log(ctx1);
    // var yearChart = new Chart(ctx1, {
    //     type: 'line',
    //     data: [1, 2]
    // });
    // var ctx2 = document.getElementById("week-chart");
    // console.log(ctx2);
    // var weekChart = new Chart("year-chart", {
    //     type: 'line',
    //     data: [3, 4]
    // });
function addData(chart, label, data) {
    // chart.data.labels.push(label);
    // chart.data.datasets.forEach((dataset) => {
    //     dataset.data.push(data);
    // });
    // chart.update();
}

function removeData(chart) {
    // chart.data.labels.pop();
    // chart.data.datasets.forEach((dataset) => {
    //     dataset.data.pop();
    // });
    // chart.update();
}


var app = new Vue({
    el: '#mainwrap',
    data: {
        search: 'Sh',
        books: [],
        selected: null,
        selected_data: null
    },
    computed: {
        results: function(){
            var books = this.books;
            var query = this.search;
            var data = [];
            for (var i = 0; i < books.length; i++) {
                if(books[i].toLowerCase().indexOf(query.toLowerCase()) > -1){
                    data.push({
                        id: i,
                        name: books[i]
                    });
                }
            }
            return data;
        }
    },
    methods: {
        select: function(el){
            this.selected_data = null;
            this.selected = el;
            this.load_data(el.id);
        },
        goToSearch: function(){
            this.selected_data = null;
            this.selected = null;
        },
        load_data: function(id){
            var that = this;
            get("/api/book?id="+id, function(data){
                that.selected_data = data;
                var N = 52;
                var arr = Array.apply(null, {length: N}).map(Number.call, Number);
                new Chart("year-chart", {
                    type: 'line',
                    data: {
                        labels: arr,
                        datasets:[{
                            label: "Rental time",
                            data: data["year"],
                            backgroundColor: "RGBA(75, 192, 192, 0.2)",
                            borderColor: "#4BC0C0",
                        }]
                    },
                    options: {
                        responsive: false,
                        scales: {
                            yAxes: [{
                                ticks: {
                                    beginAtZero:true,
                                    steps: 50,
                                    stepValue: 1,
                                    max: 50
                                }
                            }]
                        }
                    }
                });
                N = 7;
                arr = Array.apply(null, {length: N}).map(Number.call, Number)
                new Chart("week-chart", {
                    type: 'line',
                    data: {
                        labels: arr,
                        datasets:[{
                            label: "Rental time",
                            data: data["week"],
                            backgroundColor: "rgba(255, 99, 132, 0.2)",
                            borderColor: "rgba(255,99,132,1)",
                        }]
                    },
                    options: {
                        responsive: false,
                        scales: {
                            yAxes: [{
                                ticks: {
                                    beginAtZero:true,
                                    steps: 50,
                                    stepValue: 1,
                                    max: 50
                                }
                            }]
                        }
                    }
                });
            });
        }
    },
    created: function(){
        var that = this;
        get("/api/books", function(data){
            that.books = data;
        });
    }
});

function get(url, sucess){
    var request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.onload = function() {
        if (request.status >= 200 && request.status < 400) {
            var data = JSON.parse(request.responseText);
            sucess(data);
        } else {

        }
    };
    request.onerror = function() {};
    request.send();
}
