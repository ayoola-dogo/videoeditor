from django import forms
import numpy as np
import re

VALUES = [
    ('48000', '48000'),
    ('41000', '41000'),
    ]

PLAYBACK = np.arange(0.5, 3, 0.5)
PLAYBACK = zip(PLAYBACK.tolist(), PLAYBACK.tolist())


class InputParm(forms.Form):
    frame_rate = forms.IntegerField(widget=forms.Select(choices=VALUES), initial=48000)
    frame_quality = forms.IntegerField(initial=3)
    frame_margin = forms.IntegerField(initial=5)
    playback_speed = forms.FloatField(widget=forms.Select(choices=PLAYBACK), initial=1)
    silent_speed = forms.IntegerField(initial=100)
    silence_threshold = forms.FloatField(initial=0.03)

    def __init__(self, *args, **kwargs):
        super(InputParm, self).__init__(*args, **kwargs)  # Call to ModelForm constructor
        self.fields['frame_quality'].widget.attrs['min'] = 1
        self.fields['frame_margin'].widget.attrs['min'] = 0
        self.fields['silent_speed'].widget.attrs['min'] = 1
        self.fields['silence_threshold'].widget.attrs['min'] = 0

    def clean_frame_quality(self):
        frame_quality = self.cleaned_data['frame_quality']
        if 1 <= frame_quality <= 31:
            return frame_quality
        else:
            raise forms.ValidationError('Frame Quality Value must be in this range (1 to 31)')

    def clean_frame_margin(self):
        frame_margin = self.cleaned_data['frame_margin']
        if 0 <= frame_margin <= 120:
            return frame_margin
        else:
            raise forms.ValidationError('Frame Margin Value must be in this range (0 to 120)')

    def clean_playback_speed(self):
        playback_speed = self.cleaned_data['playback_speed']
        if 0.5 <= playback_speed <= 2.25:
            return playback_speed
        else:
            raise forms.ValidationError('Playback Speed Value must be in this range (0.5 to 2.25)')

    def clean_silent_speed(self):
        silent_speed = self.cleaned_data['silent_speed']
        if 1 <= silent_speed <= 100:
            return silent_speed
        else:
            raise forms.ValidationError('Silent Speed Value must be in this range (1 to 100)')

    def clean_silence_threshold(self):
        silence_threshold = self.cleaned_data['silence_threshold']
        if 0 <= silence_threshold <= 1:
            return silence_threshold
        else:
            raise forms.ValidationError('Silence Threshold Value must be in this range (0 to 1)')


QUALITY = [
    ('Default', 'Default'),
    ('1080p', '1080p'),
    ('720p', '720p'),
    ('480p', '480p'),
    ('360p', '360p'),
    ('240p', '240p'),
    ('144p', '144p')
]


class UrlForm(forms.Form):
    paste_URL = forms.URLField(widget=forms.TextInput(attrs={'rows': 1, 'cols': 100}), required=True)
    quality = forms.CharField(widget=forms.Select(choices=QUALITY), initial="Default")

    def clean_paste_URL(self):
        pattern = re.compile(r'.+youtube\.com/watch\?.+')
        paste_url = self.cleaned_data['paste_URL']
        if pattern.match(paste_url):
            print(paste_url)
            return paste_url
        else:
            raise forms.ValidationError('Please enter a youtube video')


def validate_file_extension(value):
    import os
    from django.core.exceptions import ValidationError
    ext = os.path.splitext(value.name)[1]
    valid_extensions = ['.mp4', '.mov', '.mpeg', '.wmv']
    if not ext.lower() in valid_extensions:
        raise ValidationError('Unsupported file extension.')


class UploadVideoForm(forms.Form):
    title = forms.CharField(max_length=70, required=True)
    file = forms.FileField(required=True, validators=[validate_file_extension])


class SubmitForm(forms.Form):
    origin_video_size = forms.CharField(required=False)
    origin_video_length = forms.CharField(required=False)
    new_video_size = forms.CharField(required=False)
    new_video_length = forms.CharField(required=False)
