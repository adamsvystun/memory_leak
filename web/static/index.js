var app6 = new Vue({
    el: '#mainwrap',
    data: {
        search: 'Hello Vue!',
        books: []
    },
    computed: {
        results: function(){
            var books = this.books;
            var query = this.search;
            var data = [];
            data = books.filter(function(book){
                return book.toLowerCase().indexOf(query.toLowerCase()) > -1
            });
            return data;
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
