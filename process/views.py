from .forms import InputParm, UrlForm, SubmitForm, UploadVideoForm
from django.urls import reverse
from django.shortcuts import render, HttpResponseRedirect
import os
from django.conf import settings
from django.http import HttpResponse, Http404
import re


# Create your views here.
def home(request):
    template = 'process/home.html'
    context = {}
    return render(request, template, context)


def choose_file(request):
    if request.method == 'POST':
        video_form = UploadVideoForm(request.POST, request.FILES)
        form = InputParm(request.POST)
        if form.is_valid() and video_form.is_valid():
            cd = form.cleaned_data
            cd2 = video_form.cleaned_data
            ext = os.path.splitext(str(request.FILES['file']))[1]
            with open('static/media/video_input/{}{}'.format(cd2['title'], ext), 'wb+') as destination:
                for chunk in request.FILES['file'].chunks():
                    destination.write(chunk)
            return HttpResponseRedirect(reverse("progress", kwargs={'url': " ",
                                                                    'fr': cd['frame_rate'],
                                                                    'fq': cd['frame_quality'],
                                                                    'fm': cd['frame_margin'],
                                                                    'ps': cd['playback_speed'],
                                                                    'ss': cd['silent_speed'],
                                                                    'st': cd['silence_threshold'],
                                                                    'qlty': "Default"}))
        else:
            template = 'process/choose_file.html'
            context = {'form': form, 'video_form': video_form}
            return render(request, template, context)
    else:
        form = InputParm()
        video_form = UploadVideoForm()
        template = 'process/choose_file.html'
        context = {'form': form, 'video_form': video_form}
        return render(request, template, context)


def video_url(request):
    if request.method == 'POST':
        form = InputParm(request.POST)
        url_form = UrlForm(request.POST)
        for k in request.POST:
            print(k)
        if form.is_valid() and url_form.is_valid():
            cd = form.cleaned_data
            cd2 = url_form.cleaned_data
            return HttpResponseRedirect(reverse("progress", kwargs={'url': cd2['paste_URL'],
                                                                      'fr': cd['frame_rate'],
                                                                      'fq': cd['frame_quality'],
                                                                      'fm': cd['frame_margin'],
                                                                      'ps': cd['playback_speed'],
                                                                      'ss': cd['silent_speed'],
                                                                      'st': cd['silence_threshold'], 'qlty': cd2['quality']}))
        else:
            template = 'process/video_url.html'
            context = {'form': form, 'url_form': url_form}
            return render(request, template, context)
    else:
        form = InputParm()
        url_form = UrlForm()
        template = 'process/video_url.html'
        context = {'form': form, 'url_form': url_form}
        return render(request, template, context)


def progress(request, url, fr, fq, fm, ps, ss, st, qlty):
    template = "process/progress.html"
    submit_form = SubmitForm()
    context = {'submit_form': submit_form}
    return render(request, template, context)


def download(request):
    dir = os.path.join(settings.MEDIA_ROOT, "process\\Youtube_tmp")
    pattern = re.compile(r'[-.\\/\s\w]+_videocuts_[-.\\/\s\w]+')
    for file in os.listdir(dir):
        if pattern.match(str(file)):
            path = "process\\Youtube_tmp\\" + file
            file_path = os.path.join(settings.MEDIA_ROOT, path)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as fh:
                    response = HttpResponse(fh.read(), content_type="video/mp4")
                    response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
                    return response
            raise Http404


def download_page(request):
    dir = os.path.join(settings.MEDIA_ROOT, "process\\Youtube_tmp")
    pattern = re.compile(r'[-.\\/\s\w]+_videocuts_[-.\\/\s\w]+')
    for file in os.listdir(dir):
        if pattern.match(str(file)):
            path = "process\\Youtube_tmp\\" + file
            file_path = os.path.join(settings.MEDIA_ROOT, path)
            if os.path.exists(file_path):
                template = 'process/download.html'
                context = {}
                return render(request, template, context)
            raise Http404
