import os, signal, time, prctl

class CleanChildProcesses:
    """
    Calls some function that presumably spawns a number of subprocesses.
    Makes sure that they are cleaned up on exit.

    with CleanChildProcesses() as ccp:
        res = ccp.run(func)
    """
    def __init__(self, time_to_die=5, reap_pgroup=True, foreground=False):
        """
        :param time_to_die: Number of seconds to give processes to shut down
            nicely. (how long to wait between SIGINT and SIGKILL)
        :param reap_pgroup: Whether or not to reap all zombies from the process
            group. Will make return codes unaccessible outside of the with
            scope.
        :param foreground:
        """
        self.time_to_die = time_to_die  # how long to give children to die before SIGKILL
        self.reap_pgroup = reap_pgroup
        self.foreground = foreground  # If user wants to receive Ctrl-C
        self.is_foreground = False
        self.SIGNALS = (signal.SIGHUP, signal.SIGTERM, signal.SIGABRT,
                        signal.SIGALRM, signal.SIGPIPE, signal.SIGINT)
        self.is_stopped = True  # only call stop once (catch signal xor exiting 'with')

    def _should_become_foreground(self):
        """
        Check if the user requested to become the foreground process and it
        is possible to becoe the foreground process.
        """
        if not self.foreground:
            return False
        try:
            fd = os.open(os.ctermid(), os.O_RDWR)
        except OSError:
            # Happens if process not run from terminal (tty, pty)
            return False

        os.close(fd)
        return True

    def _maybe_become_foreground(self):
        """
        Become the foreground process if the user requested it and it is
        possible. Allows the "Parent" process (aka this one) to get
        SIGINT (Ctrl+C). Useful if running in your terminal or from python repl.
        :return:
        """
        if self._should_become_foreground():
            hdlr = signal.signal(signal.SIGTTOU, signal.SIG_IGN)  # ignore since would cause this process to stop
            self.controlling_terminal = os.open(os.ctermid(), os.O_RDWR)
            self.orig_fore_pg = os.tcgetpgrp(self.controlling_terminal)  # sends SIGTTOU to this process
            os.tcsetpgrp(self.controlling_terminal, self.childpid)
            signal.signal(signal.SIGTTOU, hdlr)
            return True
        return False


    def _signal_hdlr(self, sig, framte):
        self.__exit__(None, None, None)

    def _leave_foreground(self):
        """
        This resets the foreground process group.
        :return:
        """
        if self.is_foreground:
            os.tcsetpgrp(self.controlling_terminal, self.orig_fore_pg)
            os.close(self.controlling_terminal)
            self.is_foreground = False

    def _kill_children(self):
        """
        Sends an interrup signal to children in the pgroup they are all in.
        After allowing them a chance to exit gracefully sends a SIGKILL to
        gurantee they are all dea.
        :return:
        """
        try:
            # Interrupt the entire process group.
            os.killpg(self.childpid, signal.SIGINT)
            # Let processes end gracefully.
            time.sleep(self.time_to_die)
            # In case process gets stuck while dying, and to make sure the
            # dummy child process is dead.
            os.killpg(self.childpid, signal.SIGKILL)
        except ProcessLookupError as e:
            # No more children in process group when trying to kill, therefore
            # the goal of having no processes has been achieced and
            # this isn't really an error, even if it is weird (since the
            # dummy child process should exit until SIGKILL).
            pass

    def _reap_children(self):
        """
        Reaps the process group so as not to leave Zombie (defunct) processes.
        This assumes all processes in the pgroup have successfully been
        killed otherwise will hang.
        :return:
        """

        # reap the dummy child process. This could throw if someone has reaped
        # the dummy child process, but I think it is reasonable for that to
        # throw since no one should be touching that process.
        try:
            os.waitpid(self.childpid, 0)
        except ChildProcessError as e:
            """
            can occur if dummy child process finished and one of:
            - was reaped by another process. (Shouldn't happen, since no
                one outside of this class should touch the dummy child
                process.)
            - if parent explicitly ignored SIGCHLD
                signal.signal(signal.SIGCHLD, signal.SIG_IGN)
            - parent has the SA_NOCLDWAIT flag set.
            """
            pass

        try:
            if self.reap_pgroup:
                # Reap any zombie process in the group.
                while True:
                    # will throw ChildProcessError once all the children have
                    # been reaped and will exit that way.
                    os.waitpid(-self.childpid, 0)
        except ChildProcessError as e:
            pass

    def run(self, fn):
        """
        Runs some function from within a dummy child process in its own pgroup.
        This allows for calls us to send signals to kill the entire pgroup,
        and also allows for send a deathsig in case the subprocess resets
        its pgroup. This should handle if a subprocess also calls to
        CleanChildProcesses.
        """
        self.is_stopped = False
        """
        When running out of remote shell, SIGHUP is only sent to the session
        leader normally, the remote shell, so we need to make sure we are sent 
        SIGHUP. This also allows us not to kill ourselves with SIGKILL.
        - A process group is called orphaned when the parent of every member is 
            either in the process group or outside the session. In particular, 
            the process group of the session leader is always orphaned.
        - If termination of a process causes a process group to become orphaned, 
            and some member is stopped, then all are sent first SIGHUP and then 
            SIGCONT.
        """
        self.childpid = os.fork()  # return 0 in the child branch, and the childpid in the parent branch
        if self.childpid == 0:
            try:
                os.setpgrp()  # create new process group, become its leader
                """
                Using prctl means that if a subprocesses run from within this scope
                also uses a CleanProcessGroup, that group of processes will still get
                stopped.

                Note: I am not guranteeing this because prctl isn't in the std lib,
                and I previousely had some issues downloading it. Can't pip3 it used 
                "sudo apt-get install python3-prctl".
                """
                prctl.set_pdeathsig(signal.SIGINT)
                return fn()

                os.kill(os.getpid(), signal.SIGSTOP)  # child fork stops itself
            finally:
                os._exit(0)  # shut down without going to __exit__

        os.waitpid(self.childpid, os.WUNTRACED)  # wait until child stopped after it created the process group
        os.setpgid(0, self.childpid)  # join child's group so processes spawned under this classes scope will be within the child's group.
        self.is_foreground = self._maybe_become_foreground()

        self.exit_signals = {s: signal.signal(s, self._signal_hdlr)
                             for s in self.SIGNALS}

    def stop(self):
        try:
            for s in self.SIGNALS:
                #don't get interrupted while cleaning everything up
                signal.signal(s, signal.SIG_IGN)

            self.is_stopped = True
            self._leave_foreground()

            os.setpgrp()  # leave the child's process group so I won't get signals
            self._kill_children()  # kills off processes in childpid group.
            time.sleep(1)
            self._reap_children()  # reaps Zombie processes
        finally:
            for s, hdlr in self.exit_signals.items():
                signal.signal(s, hdlr)  # reset default handlers

    def __enter__(self):
        if not self.is_stopped:
            #TODO: This should probably just raise an error.
            # don't want to do anything on start.

    def __exit__(self, exit_type, value, traceback):
        if not self.is_stopped:
            self.stop()

class FakeCPG(object):
    def __enter__(self):
        self.origpid = os.getpid()
        print('orig =', os.getpid())
        self.childpid = os.fork()
        if self.childpid != 0:
            os.kill(os.getpid(), signal.SIGSTOP)
            time.sleep(5)
        else:
            print('child =', os.getpid())

    def __exit__(self, exit_type, value, traceback):
        print('exit =', os.getpid())
        if self.childpid == 0:
            print('child __exit__')
            os.kill(self.origpid, signal.SIGCONT)
            os.kill(os.getpid(), signal.SIGSTOP)
        else:
            print('orig __exit__')

