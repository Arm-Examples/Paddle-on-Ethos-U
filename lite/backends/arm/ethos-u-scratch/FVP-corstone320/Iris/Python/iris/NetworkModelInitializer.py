#!/usr/bin/env python
# \file   NetworkModelInitializer.py
# \brief  Prepares and establishes a connection between the Iris debugger and an Iris server.
#
# \date   Copyright ARM Limited 2020-2023 All Rights Reserved.

from __future__ import print_function
import subprocess
import re
import os
import sys
import shlex
import socket
from time import sleep

from . import debug
NetworkModel = debug.Model.NetworkModel

SERVER_WAIT_TIME_IN_SECONDS = 4

def _wait_process(proc, initial_timeout_ms=5000, timeout_ms_after_sigint=5000, timeout_ms_after_sigkill=5000):
    # The timeouts are not implemented for Python 2.7. Still works for the "good" case.
    if sys.version_info < (3, 0):
        proc.wait()
        return

    # NOTE: this mimicks IrisClient::disconnectAndWaitForChildToExit
    def __wait_process_impl(timeout, pre):
        if timeout is not None and timeout < 0:
            timeout = None
        if timeout == 0:
            return False
        pre()
        try:
            proc.wait(timeout * 1e-3)
        except subprocess.TimeoutExpired:
            return False
        return True

    if __wait_process_impl(initial_timeout_ms, lambda: None):
        return
    print("Sending SIGINT while waiting for model to exit.", file=sys.stderr)
    if __wait_process_impl(timeout_ms_after_sigint, lambda: proc.send_signal(subprocess.signal.SIGINT)):
        return
    print("Sending SIGKILL while waiting for model to exit.", file=sys.stderr)
    if __wait_process_impl(timeout_ms_after_sigkill, lambda: proc.kill()):
        return
    try:
        proc.wait(0.0)
    except subprocess.TimeoutExpired:
        raise RuntimeError("model did not exit within the allotted timeout")


class NetworkModelInitializer(object):
    """
    The NetworkModelInitializer class represents an established or pending connection
    between an Iris Model Debugger, accessible via the class NetworkModel, and an Iris
    server which is embedded either within an ISIM or another simulation using an ISIM as
    a library.
    You should use the NetworkModelFactory class below to create an instance of this class.

    Once the class is created you can use it in two ways:

    1: network_model below is an instance of NetworkModel, all resources are automatically
       deallocated at the end of the with statement context.

       with NetworkModelFactory.CreateNetworkToHost(host, port) as network_model:
           network_model.get_targets()

    2: network_model below is an instance of NetworkModel, all resource are NOT automatically
       deallocatted so you  need to handle exception and force deallocation manually.

       network_model_initializer = NetworkModelFactory.CreateNetworkToHost(host, port)
       network_model = network_model_initializer.start()
       try:
           network_model.get_targets()
       finally:
           network_model_initializer.close()

    A full working example is in the Python/Example folder
    """

    def __init__(self, server_startup_command = None, host = 'localhost', port = None, timeout_in_ms = 20000, synchronous = False, env = None, verbose = False):
        self.server_startup_command = server_startup_command
        self.process = None
        self.host = host
        self.port = port
        self.timeout_in_ms = timeout_in_ms
        self.synchronous = synchronous
        self.fm_env = env
        self.verbose = verbose
        self.network_model = None

    def __get_port(self):
        while self.process.returncode is None:
            match = re.match('Iris server started listening to port ([0-9]+)',
                             self.process.stdout.readline().decode())
            if match is not None:
                return int(match.group(1))
            self.process.poll()

        raise RuntimeError('isim exited without printing a port number\n returncode: {}'.format(self.process.returncode))

    def __start_server(self):
        if self.server_startup_command is not None:
            self.process = subprocess.Popen(self.server_startup_command,
                                        env = self.fm_env,
                                        stdin = subprocess.PIPE,
                                        stderr = subprocess.PIPE,
                                        stdout = subprocess.PIPE)

            # Give some time to the server to print the port
            sleep(SERVER_WAIT_TIME_IN_SECONDS)
            self.port = self.__get_port()

    def __start_client(self):
        self.network_model = debug.Model.NewNetworkModel(
               self.host, self.port, self.timeout_in_ms, synchronous=self.synchronous, verbose=self.verbose
               )

    def close(self, initial_timeout_ms=5000, timeout_ms_after_sigint=5000, timeout_ms_after_sigkill=5000):
        """
        Deallocate the Iris server process if one was created previously

        Release and shutdown the model (child) using Iris
        If the model cannot be accessed using Iris, kill the model

        Wait at most initial_timeout_ms until the child exits.
        If the child did not exit by then, send a SIGINT and wait for timeout_after_sigint until the child exits.
        If the child did not exit by then, send a SIGKILL and wait for timeout_after_sigkill until the child exits.
        If the child did not exit by then, a RuntimeError exception is thrown.
        If initial_timeout_ms is 0, do not wait and continue with SIGINT.
        If timeout_after_sigint is 0, do not issue a SIGINT and continue with SIGKILL
        If timeout_after_sigkill is 0, do not issue a SIGKILL perform a non-blocking wait
        If the wait does not succeed, raise a RuntimeError
        If any of the timeouts is < 0, wait indefinitely at the specified step.
        """
        # If we started the model, shut it down as we are about to kill the
        # process anyway. This should allow for a clean exit without requiring
        # issuing either SIGINT or SIGKILL.
        if self.network_model is not None:
            self.network_model.release(shutdown=(self.process is not None))

        # but if network_model is None, it means we do not have access to the model (could be because of failed connection)
        # therefore kill the model subprocess directly
        elif self.process is not None:
            self.process.kill()

        if self.process is not None:
            _wait_process(self.process, initial_timeout_ms, timeout_ms_after_sigint, timeout_ms_after_sigkill)

        self.network_model = None
        self.process = None

    def start(self):
        """
        Start the Iris server (if necessary) and connects the Iris Debugger client to the server
        """
        try:
            self.__start_server()
            self.__start_client()
        except:
            self.close()
            raise
        return self.network_model

    def __enter__(self):
        return self.start()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class SocketpairPool:
    """Pool of socket pairs used to connect to fork()ed models.

    All socket pairs needed by later forks of any of the model processes
    need to be created upfront in the (top-level parent) client process
    because they are transported into the model child processes by file
    descriptor inheritance.
    """
    def __init__(self, num_socketpairs):
        """Constructor. Allocate socketpair pool.
        """
        self.socketpairs = [socket.socketpair() for i in range(num_socketpairs)]
        self.freeIndexes = [i for i in range(len(self.socketpairs))]

    def destroy(self):
        """Close all sockets in the pool.
        """
        for (lhs, rhs) in self.socketpairs:
            lhs.close()
            rhs.close()

    def allocSocketpair(self):
        """Allocate and return one socketpair.
        """
        if len(self.freeIndexes) == 0:
            raise RuntimeError("Out of socketpairs for clients! Increase max_fork_concurrency in LocalNetworkModelInitializer() to at least {}.".format(len(self.socketpairs) - 1 + 1))
        return self.socketpairs[self.freeIndexes.pop(0)]

    def releaseSocketpair(self, lhs):
        """Release a socketpair which was previously allocated using allocSocketpair().
        """
        for i in range(len(self.socketpairs)):
            if lhs == self.socketpairs[i][0]:
                self.freeIndexes.append(i)
                self.freeIndexes.sort()
                return
        raise RuntimeError("SocketpairPool: releaseSocketpair(): Unknown socketpair.")

    def getRhsFilenoList(self):
        """Get list of file descriptor nunbers of the rhs sockets.
        """
        return  [rhs.fileno() for (lhs, rhs) in self.socketpairs]


class LocalNetworkModelInitializer(object):
    def __init__(self, server_startup_command, timeout_in_ms, verbose=False, synchronous=False, stdout=None, stderr=None, env=None, max_fork_concurrency = 1, client_name = "client.iris_debug"):
        self.__command = server_startup_command
        self.__timeout = timeout_in_ms
        self.__verbose = verbose
        self.__process = None
        self.__synchronous = synchronous
        self.__model = None
        self.__stdout = stdout
        self.__stderr = stderr
        self.__fm_env = env
        self.__max_fork_concurrency = max_fork_concurrency
        self.__client_name = client_name
        if stdout == subprocess.PIPE or stderr == subprocess.PIPE:
            raise ValueError("stdout and stderr do not support the PIPE value")

    def start(self):
        if os.name == 'nt':
            raise NotImplementedError()
        else:
            # Create pool of socket pairs: One pair for this child process and one for each concurrent fork.
            socketpairpool = SocketpairPool(self.__max_fork_concurrency + 1)
            (lhs, rhs) = socketpairpool.allocSocketpair()
            command = self.__command[:]
            command.append("--iris-connect")
            command.append("socketfd={}".format(rhs.fileno()))
            if self.__verbose:
                print("Starting command: " + " ".join(command))
            try:
                if sys.version_info < (3, 0):
                    self.__process = subprocess.Popen(command, stdout=self.__stdout, stderr=self.__stderr, env=self.__fm_env)
                else:
                    self.__process = subprocess.Popen(command, stdout=self.__stdout, stderr=self.__stderr, pass_fds=socketpairpool.getRhsFilenoList(), env=self.__fm_env)
            except Exception:
                socketpairpool.destroy()
                raise
            try:
                self.__model = debug.Model.NewUnixDomainSocketModel(
                        lhs, timeoutInMs=self.__timeout,
                        verbose=self.__verbose, synchronous=self.__synchronous, socketpairpool=socketpairpool, client_name=self.__client_name)
                return self.__model
            except Exception:
                socketpairpool.destroy()
                self.close()
                raise

    def close(self, initial_timeout_ms=5000, timeout_ms_after_sigint=5000, timeout_ms_after_sigkill=5000):
        # We are the only connected client and disconnecting will automatically shut down the model.
        # Thus we pass shutdown=False in order to avoid a redundant and racy simulation_requestShutdown()
        # call from the client. Calling with shutdown=True works ok as well but is redundant and creates
        # nondeterministic shutdown logs (due to the race between the simulation_requestShutdown() from the client
        # and from the server (upon detecting the disconnect).
        if self.__model is not None:
            self.__model.release(shutdown=False)
        if self.__process is not None:
            _wait_process(self.__process, initial_timeout_ms, timeout_ms_after_sigint, timeout_ms_after_sigkill)

        self.__model = None
        self.__process = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


def split_command_line(cmd):
    """Split space-separated command line into a list of executable and args.

    This heuristically supports executables paths with spaces in them. If the
    executable file is not found [cmd] is returned.
    """
    fields = cmd.split(" ")
    if len(fields) == 0:
        return fields
    executable = fields.pop(0)
    while (not os.path.isfile(executable)) and (len(fields) > 0):
        executable += " " + fields.pop(0)
    return [executable] + fields


class NetworkModelFactory:
    """
    The NetworkModelFactory class allows the creation of NetworkModelInitializers. It contains only class methods.
    """

    @classmethod
    def CreateNetworkFromIsim(cls, isim_filename, parameters = None, timeout_in_ms = 20000, verbose=False):
        """
        Create a network initializer to an isim yet to be started
        """

        parameters = parameters or {}
        isim_startup_command = [isim_filename, '-I', '-p']  # Start Iris server and print the port

        for param, value in parameters.items():
            isim_startup_command += ['-C', '{}={}'.format(param, value)]

        return NetworkModelInitializer(server_startup_command = isim_startup_command, timeout_in_ms = timeout_in_ms, verbose = verbose)

    @classmethod
    def CreateLocalFromIsim(cls, isim_filename, parameters=dict(), timeout_in_ms=20000, verbose=False, xargs=None, synchronous=False, stdout=None, stderr=None, env=None, max_fork_concurrency = 1, client_name="client.iris_debug"):
        """
        Create a network initializer to an isim yet to be started using 1:1
        network communication, no TCP server are started and the isim will
        automatically shut down would this process terminate unexpectedly.

        :param isim_filename Path of the isim executable, or alternatively, if xargs is None, a space-separated isim command line with isim options.

        :param xargs
            A list of additional arguments to pass to the model.
            E.g., ["-C", "cpu.NUM_CORES=1"].

        :param synchronous
            Whether to instantiate a SyncModel or an AsyncModel. Threads are
            used either way for the communication.

        :param stdout
        :param stderr
            Where to redirect the sub-process outputs. E.g., to redirect the
            model's stderr to stdout, the special value
            stderr=subprocess.STDOUT can be used. To redirect stdout to stderr,
            stdout=sys.stderr.buffer can be used.

        :param max_fork_concurrency
            Maximum number of fork()ed processes which will live concurrently to the main simulation process at any point in time.
            The default of 1 is sufficient for any sequence of forks, where each fork terminates before the next fork is created.

        :param client_name
            Instance name of the Python client. Should start with "client.".
        """
        if xargs == None:
            isim_startup_command = split_command_line(isim_filename)
        else:
            isim_startup_command = [isim_filename] + xargs

        for k, v in parameters.items():
            isim_startup_command.append("-C")
            isim_startup_command.append("{}={}".format(k,v))

        return LocalNetworkModelInitializer(server_startup_command=isim_startup_command, timeout_in_ms=timeout_in_ms, verbose=verbose, synchronous=synchronous, stdout=stdout, stderr=stderr, env=env, max_fork_concurrency=max_fork_concurrency, client_name=client_name)

    @classmethod
    def CreateNetworkFromLibrary(cls, simulation_command, library_filename, parameters = None, timeout_in_ms = 20000, env = None):
        """
        Create a network initializer to a simulation application that uses an isim as a library and is not yet started
        """

        parameters = parameters or {}
        simulation_startup_command = [simulation_command, library_filename, '-I', '-p']  # Start Iris server and print the port

        for param, value in parameters.items():
            simulation_startup_command += ['-C', '{}={}'.format(param, value)]

        return NetworkModelInitializer(server_startup_command = simulation_startup_command, timeout_in_ms = timeout_in_ms, env = env)


    @classmethod
    def CreateNetworkFromCommand(cls, command_line, timeout_in_ms = 20000, env = None):
        """
        Create a network initializer to an Iris server to be started by the input command line
        """

        return NetworkModelInitializer(server_startup_command = shlex.split(command_line), timeout_in_ms = timeout_in_ms, env = env)


    @classmethod
    def CreateNetworkToHost(cls, hostname, port, timeout_in_ms = 20000, synchronous = False, verbose = False, env = None):
        """
        Create a network initializer to an iris server which was already started and is accessible at the given hostname and port
        """

        return NetworkModelInitializer(host = hostname, port = port, timeout_in_ms = timeout_in_ms, synchronous = synchronous, env = env, verbose = verbose)
