"""videoeditor URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from process.views import home, choose_file, video_url, progress, download, download_page
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('file/', choose_file, name='choose_file'),
    path('url/', video_url, name='video_url'),
    path('progress/<path:url>&fr=<int:fr>&fq=<int:fq>&fm=<int:fm>&ps=<ps>&ss=<int:ss>&st=<st>&qlty=<qlty>/',
         progress, name='progress'),
    path('download_page/', download_page, name='download_page'),
    path('download/', download, name='download')
]

urlpatterns += staticfiles_urlpatterns()

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
