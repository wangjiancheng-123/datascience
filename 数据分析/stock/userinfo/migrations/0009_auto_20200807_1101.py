# Generated by Django 2.2.12 on 2020-08-07 03:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('userinfo', '0008_auto_20181023_1419'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userinfo',
            name='last_name',
            field=models.CharField(blank=True, max_length=150, verbose_name='last name'),
        ),
    ]
