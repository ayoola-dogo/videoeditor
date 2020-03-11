from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from process.consumers import RunProcess

application = ProtocolTypeRouter({
    # (http->django views is added by default)
    'websocket': AuthMiddlewareStack(
        URLRouter(
            [
                path(r'progress/<path:url>&fr=<int:fr>&fq=<int:fq>&fm=<int:fm>&ps=<ps>&ss=<int:ss>&st=<st>&qlty=<qlty>/',
                     RunProcess),
            ]
        )
    ),
})
