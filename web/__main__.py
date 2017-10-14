import json
from random import randint

from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.response import FileResponse
from web.db import BOOKS
from web.nn_web import query

def random_arr(n):
    arr = []
    for i in range(n):
        arr.append(randint(0, 50))
    return arr


def book_api(request):
    pk = request.params.get('id', 0)
    data = query(int(pk))
    return Response(json.dumps(data))

def books_api(request):
    return Response(json.dumps(BOOKS))

def hello_world(request):
    return FileResponse(
        "web/index.html",
        request=request,
        content_type='text/html'
    )

def index_css(request):
    return FileResponse(
        "web/static/index.css",
        request=request,
        content_type='text/css'
    )

def index_js(request):
    return FileResponse(
        "web/static/index.js",
        request=request,
        content_type='text/javascript'
    )

if __name__ == '__main__':
    with Configurator() as config:
        config.add_route('hello', '/')
        config.add_route('css', '/index.css')
        config.add_route('js', '/index.js')
        config.add_route('books_api', '/api/books')
        config.add_route('book_api', '/api/book')
        config.add_view(hello_world, route_name='hello')
        config.add_view(index_css, route_name='css')
        config.add_view(index_js, route_name='js')
        config.add_view(books_api, route_name='books_api')
        config.add_view(book_api, route_name='book_api')
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()
