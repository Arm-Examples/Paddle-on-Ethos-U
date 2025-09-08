#!/usr/bin/env python

#
# Copyright (c) 2019-2020,2022 Arm Limited. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import json
import os
import subprocess
import sys

def print_args(args, **kwargs):
    cwd = kwargs['cwd']

    if isinstance(args, list):
        args = ' '.join(args)

    print('%s$ %s' % (cwd, args))

def check_call(args, **kwargs):
    print_args(args, **kwargs)
    return subprocess.check_call(args, **kwargs)

def check_output(args, **kwargs):
    print_args(args, **kwargs)
    return subprocess.check_output(args, **kwargs)

###############################################################################
# Git class
###############################################################################

class Git(object):
    def __init__(self, pwd, path, name, fetchurl, pushurl=None, revision='master'):
        self.pwd = pwd
        self.path = path
        self.name = name
        self.absolutepath = os.path.join(pwd, path)
        self.fetchurl = fetchurl
        self.pushurl = pushurl
        self.revision = revision

    def checkout_and_update(self):
        self.init()
        self.remote_add(self.name, self.revision, self.fetchurl, self.pushurl)
        self.fetch(self.name, self.revision)
        self.checkout(self.name, 'FETCH_HEAD')

    def init(self):
        if not os.path.exists(self.absolutepath):
            os.makedirs(self.absolutepath)

        if not os.path.exists(os.path.join(self.absolutepath, '.git')):
            check_output(['git', 'init'], cwd=self.absolutepath)

    def remote_add(self, name, revision, fetchurl, pushurl):
        remotes = self.__get_remotes()

        if name in remotes:
            if fetchurl != remotes[name]['fetch']:
                raise Exception("Fetch url '%s' from repository for remote '%s' does not match fetch url '%s' from manifest." % (fetchurl, name, remotes[name]['fetch']))

            if pushurl not in (None, remotes[name]['push']):
                raise Exception("Push url '%s' from repository for remote '%s' does not match push url '%s' from manifest." % (pushurl, name, remotes[name]['push']))
        else:
            check_output(['git', 'remote', 'add', '-m', revision, name, fetchurl], cwd=self.absolutepath)

            if pushurl:
                check_output(['git', 'remote', 'set-url', '--add', '--push', name, pushurl], cwd=self.absolutepath)

    def fetch(self, name, revision):
        check_output(['git', 'fetch', name, revision], cwd=self.absolutepath)

    def checkout(self, name, revision):
        rev = self.__get_rev(name, revision)
        check_output(['git', 'checkout', rev], stderr=subprocess.STDOUT, cwd=self.absolutepath)

    def get_dict(self, sha1):
        data = {}
        data['path'] = self.path
        data['name'] = self.name
        data['fetchurl'] = self.fetchurl

        if self.pushurl:
            data['pushurl'] = self.pushurl

        if sha1:
            data['revision'] = self.__get_rev(self.name, self.revision)
        else:
            data['revision'] = self.revision

        return data

    def __get_rev(self, name, revision):
        try:
            rev = check_output(['git', 'rev-parse', name + '/' + revision], stderr=subprocess.STDOUT, cwd=self.absolutepath)
        except:
            rev = check_output(['git', 'rev-parse', revision], cwd=self.absolutepath)

        return rev.decode('utf-8').strip()

    def __get_remotes(self):
        remotes = {}
        for remote in check_output(['git', 'remote'], cwd=self.absolutepath).decode('utf-8').splitlines():
            fetch = check_output(['git', 'remote', 'get-url', remote], cwd=self.absolutepath).decode('utf-8').strip()
            push = check_output(['git', 'remote', 'get-url', '--push', remote], cwd=self.absolutepath).decode('utf-8').strip()
            remotes[remote] = { 'fetch': fetch, 'push': push }

        return remotes

###############################################################################
# Externals class
###############################################################################

class Externals:
    def __init__(self, config):
        self.externals = []
        self.load_config(config)

    def load_config(self, config):
        self.pwd = os.path.dirname(os.path.realpath(config))

        with open(config, 'r') as fp:
            data = json.load(fp)

        for ext in data['externals']:
            git = Git(self.pwd, ext['path'], ext.get('name', 'origin'), ext['fetchurl'], ext.get('pushurl', None), ext['revision'])
            self.externals.append(git)

    def fetch(self):
        for ext in self.externals:
            ext.checkout_and_update()

    def get_dict(self, sha1):
        data = { 'externals': [] }
        for ext in self.externals:
            data['externals'].append(ext.get_dict(sha1))

        return data

    def dump(self, fhandle, sha1):
        fhandle.write(json.dumps(self.get_dict(sha1), sort_keys=False, separators=(',', ': '), indent=4))

###############################################################################
# Handle functions
###############################################################################

def handle_fetch(args):
    externals = Externals(args.configuration)
    externals.fetch()

def handle_dump(args):
    externals = Externals(args.configuration)
    externals.dump(sys.stdout, args.sha1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch external repositories.')
    parser.add_argument('-c', '--configuration', default='externals.json', help='Externals configuration file')
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser('fetch', description='Fetch external repositories.')
    subparser.set_defaults(func=handle_fetch)

    subparser = subparsers.add_parser('dump', description='Dump configuration.')
    subparser.set_defaults(func=handle_dump)
    subparser.add_argument('-s', '--sha1', action='store_true', help='Replace revision with current SHA-1')

    args = parser.parse_args()

    if 'func' not in args:
        parser.print_help()
        sys.exit(2)

    sys.exit(args.func(args))

