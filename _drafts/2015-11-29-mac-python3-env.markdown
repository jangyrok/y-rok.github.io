---
layout: post
title:  "[Mac]Python3 개발 환경 구축 - Python, virtualenv"
date:   2017-11-29 00:00:00
author: y-rok
categories: python
tags: python virtualenv
---


## Python3 설치

`Homebrew`를 이용하여 `python3`를 설치한다.

```bash
❯❯❯ brew install python3
```

이 후 설치가 정상적으로 되었는지 확인한다.

```bash
❯❯❯ python3 --version
Python 3.7.1
```

하지만, `python3` 명령어 대신 `python`을 사용할 경우 mac에 기본적으로 설치 되어 있는 `python2`가 사용된다.

```bash
❯❯❯ python --version
Python 2.7.10
```

따라서, `~/.bashrc` 파일에 다음의 내용을 추가한다.
(zshell을 사용하는 경우 `~/.zshrc`에 내용 추가)
```
#custom code
alias python='/usr/local/bin/python3'
```

이 후, Terminal을 다시 실행하면 `python` 명령어 사용시 설치된 `python3`가 사용됨을 알 수 있다.

```bash
❯❯❯ python --version
Python 3.7.1
```

# virtualenv를 활용하여 가상환경 설정

파이썬 작업을 하다 보면 `pip3`를 활용하여 프로젝트 별로 package를 설치하는 일이 상당히 많다. package를 Global하게 설치할 수도 있겠지만 프로젝트 별로 따로 package를 관리하고 싶다면 `virtualenv`를 사용하면 된다. 개념이 조금 햇갈릴 수 있지만... 단순히 `pip3`로 package 설치 시 package들을 따로 특정 폴더에 저장하고 이를 활용하는 방식이다. (자세한 설명은 [이곳](https://medium.com/@dan_kim/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%B4%88%EC%8B%AC%EC%9E%90%EB%A5%BC-%EC%9C%84%ED%95%9C-pip-%EA%B7%B8%EB%A6%AC%EA%B3%A0-virtualenv-%EC%86%8C%EA%B0%9C-a53512fab3c2)을 참고하자.)

`pip3`를 활용하여 `virtualenv`를 설치하자.
```bash
❯❯❯ pip3 install virtualenv
```

`virtualenv`를 설치 했다면 다음 명령어를 통해 가상환경을 생성할 수 있다. (여기에서 가상환경이란 따로 package 관리하는 폴더를 말함)

```bash
❯❯❯ virtualenv --system-site-packages -p python3 ./env
```

- --system-site-packages = 가상 환경에서도 global package에 접근 가능 하도록 설정
- -p = 가상환경의 python interpreter 설정
- 가상황경 폴더 이름 설정 (ex) ./env)


이제 다음 명령어를 통해 가상환경을 실행할 수 있다. (설정된 python interpreter를 사용, 특정 폴더의 package들 활용)

```bash
❯❯❯ source ./env/bin/activate
```

만약, `zshell`을 사용한다면 terminal에 "env"가 표시되어 가상환경일 설정되어 있다는 것을 확인할 수 있다.

![](/assets/img/2015-11-29-mac-python3-env/virtualenv_activated.png)

이 후, pip3로 package 설치 시 가상환경 폴더 내에 `lib/python3.7/site-packages`에 package들이 설치될 것이다.

가상환경 내에서의 작업을 종료하고 싶다면 다음과 같이 하면 된다.
```bash
❯❯❯ deactivate
```
