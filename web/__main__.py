from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.response import FileResponse

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
        config.add_view(hello_world, route_name='hello')
        config.add_view(index_css, route_name='css')
        config.add_view(index_js, route_name='js')
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()
