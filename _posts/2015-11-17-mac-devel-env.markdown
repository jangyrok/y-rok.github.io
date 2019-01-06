---
layout: post
title:  "Mac 개발 환경 설정 - Homebrew, Tmux, zshell 설치 하기"
date:   2017-11-17 00:00:00
author: y-rok
categories: Mac
tags: zshell homebrew tmux
---


**_(해당 Post의 프로그램 설치는 Mac os - Mojave 10.14 기반으로 수행하였습니다.)_**

이번 Post에서는 기본적으로 `Mac` 개발에 필요한 요소들의 설치 과정을 설명한다. 설치 목록은 다음과 같다. 

#### Homebrew
- Linux의 yum, apt-get과 유사하게 Mac에서 가장 많이 쓰이는 Package 관리자
+

#### Tmux 
- 1개의 윈도우 안에서 여러개의 terminal을 화면 분할 등을 통해 활용할 수 있는 Terminal Multiplexer 

#### Dotfiles & zshell
- Unix file system에서의 application, service, tool 등의 환경 설정 파일들 (ex) .bash_profile)
- zshell 기반 다음 [링크](https://github.com/wookayin/dotfiles)의 `dotfiles`로 설정한다. 


## Homebrew 설치

다음 명령어를 통해 `Homebrew`를 설치한다.

```bash
>>> /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

... (생략) ...

Downloading Command Line Tools (macOS Mojave version 10.14) for Xcode
Downloaded Command Line Tools (macOS Mojave version 10.14) for Xcode
Installing Command Line Tools (macOS Mojave version 10.14) for Xcode
Done with Command Line Tools (macOS Mojave version 10.14) for Xcode
Done.

... (생략) ...

==> Homebrew is run entirely by unpaid volunteers. Please consider donating:
  https://github.com/Homebrew/brew#donations
==> Next steps:
- Run `brew help` to get started
- Further documentation: 
    https://docs.brew.sh
```

`Homebrew` 설치 시 `Command Line Tool`도 설치 되기 때문에 `git`, `gcc` 와 같은 명령어도 추가된다. 이를 확인해 보자

```bash
>>> brew --version
Homebrew 1.8.2
```

```bash
>>> git --version
git version 2.19.0
```

## Tmux  설치

`Tmux`는 위에서 설치 된 brew를 활용하여 간단히 설치 가능하다.
**_(`Tmux`의 사용법은 [링크](https://bluesh55.github.io/2016/10/10/tmux-tutorial/)를 참고한다.)_**

```bash
>>> brew install tmux

... (생략) ...


==> tmux
Example configuration has been installed to:
  /usr/local/opt/tmux/share/tmux

Bash completion has been installed to:
  /usr/local/etc/bash_completion.d
```
## Dotfiles 설정 및 zshell 설정

[링크](https://github.com/wookayin/dotfiles)의 `dotfiles`를 clone 해온다.

```bash
>>> git clone --recursive https://github.com/wookayin/dotfiles.git ~/.dotfiles

... (생략) ... 

remote: Total 285 (delta 0), reused 0 (delta 0), pack-reused 285        
Receiving objects: 100% (285/285), 59.22 KiB | 227.00 KiB/s, done.
Resolving deltas: 100% (170/170), done.
Submodule path 'tmux/plugins/tpm/lib/tmux-test': checked out '33fa65fbfb72ba6dd106c21bf5ee6cc353ecdbb6'
Submodule path 'vim/bundle/vim-plug': checked out '46ae29985d9378391c3e1ec8a50d8229afeea084'
Submodule path 'zsh/fasd': checked out '287af2b80e0829b08dc6329b4fe8d8e5594d64b0'
Submodule path 'zsh/is_mosh': checked out '6cde1cf5d4af45b2f9bcc4267d0beca0b2b61c17'
Submodule path 'zsh/zgen': checked out 'ffd3f50addf7a0106d9ea199025eb99efbb858f4'
Submodule path 'zsh/zplug': checked out 'cd82438f89f3d17351bc78cdd424558552e3fb3c'
```

installation script를 실행한다. (중간에 sudo 권한을 위해 password)

```bash
>>> cd ~/.dotfiles && python install.py

... (생략) ...

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ You have   4 warnings or errors -- check the logs!  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
   ~/.tmux/plugins/tpm/bin/install_plugins
   # Check tmux version >= 2.3 (or use `dotfiles install tmux`)
   # Change default shell to zsh
   # Create ~/.gitconfig.secret file and check user configuration


- Please restart shell (e.g. `exec zsh`) if necessary.
- To install some packages locally (e.g. neovim, tmux), try `dotfiles install <package>`
- If you want to update dotfiles (or have any errors), try `dotfiles update`
```

Default shell을  `zshell`로 변경한다.

```bash
>>>  chsh -s `which zsh`
```

Terminal을 다시 키면 다음과 같이 zshell이 실행되는 것을 확인할 수 있다.

![](/assets/img/2015-11-17-mac-devel-env/zsh.png)