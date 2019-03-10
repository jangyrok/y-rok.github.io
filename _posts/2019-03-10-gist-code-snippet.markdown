---
layout: post
title:  "Gist와 VS Code로 Code Snippet 관리하기"
date:   2019-03-10 00:00:00
author: y-rok
categories: 
- gist
tags: 
- gist 
- code snippet
- github
---


이번 포스트에서는 Gist와 Visual Studio Code로 Code snippet을 관리하는 방법이 대해 소개합니다.

**Gist** 는 Code snippet을 관라하기 위한 **Github**의 서비스입니다.

개발을 오래 하다 보면 여러번 같은 Code를 작성하게 될 때가 있습니다. 예를 들어, python에서 Command Line으로부터 Argument를 받아오는 Code는 매우 자주 쓰여 python 파일을 만들 떄 마다 coding 하게됩니다. 이러한 **Code snippet을 따로 서버에 관리하며 필요시 IDE에서 coding 하다가 바로 가져와 쓰면 매우 편리**한데 이에 대해 설명하고자 합니다. 

- **Gist**로 Code Snippet 관리 하기
- **VS Code**로 Gist 소스 가져오기


## Gist로 Code Snippet 관리하기

**Gist**는 **Github**의 Repository 처럼 git으로 Version 관리가 가능합니다. 단, **Github**는 Project 관리를 위한 Repository라면 **Gist**는 하나 혹은 여러개의 소스, 떄로는 전체 프로젝트 소스들의 공유하는 용도로 사용되는 Repository입니다.

[Gist](https://gist.github.com/)로 들어가면 다음과 같이 Code Snippet을 작성할 수 있습니다.

![](/assets/img/2019-03-10-gist-code-snippet/gist_create.png)

적고자 하는 파일이 여러가지인 경우 파일을 추가 할 수 있고 **Gist**에서는 **public** 혹은 **private**으로 생성할 수 있습니다.

- **public**으로 생성할 경우 [Gist Discover](https://gist.github.com/discover/)에서 검색이 가능합니다.
- **private**은 **Gist Discover**에서 검색이 불가능 하지만 해당 page의 url로 모든 User가 접근 가능합니다. 사실상, 검색만 불가능한 구조이므로 접근 권한 자체를 제한하고 싶다면 **Github**의 Repositiory를 사용해야 합니다.

![](/assets/img/2019-03-10-gist-code-snippet/gist_result.png)

생성된 [Gist Code](https://gist.github.com/y-rok/403401a9933ff36c9357edf2697a6d2c)를 보면 **comment** 기능도 있어 Code를 올리고 다른 User와 소통도 가능합니다.

## VS Code로 Gist 소스 가져오기

**Gist**에 Code snippet을 저장하고 이를 VS Code에서 필요시 가져오는 방법을 소개하고자 합니다.
(SublimeText나 Intellij IDE에서도 가능한 걸로 알고 있습니다.)

VS Code에서 **Gist** Plugin을 설치합니다.

![](/assets/img/2019-03-10-gist-code-snippet/vscode_gist_install.png)

설치 후 먼저 **Github**와 연결을 위해 Token이라는 것을 받아야합니다.

[Github Setting](https://github.com/settings/tokens)으로 들어가서 다음과 같이 Token을 만들어 줍니다.

![](/assets/img/2019-03-10-gist-code-snippet/gist_sync.png)

Token은 아래와 같이 생성됩니다. (페이지에서 나가면 다시 볼 수 없으므로 미리 복사를 합니다.)

![](/assets/img/2019-03-10-gist-code-snippet/gist_token.png)


이제 다시 VS Code로 돌아와서 **Command+Shift+p**를 눌러서 **Gist: select profile**을 통해 생성된 Token을 등록합니다.

![](/assets/img/2019-03-10-gist-code-snippet/gist_select_profile.png)

성공적으로 추가했다면, 이제 VS Code를 통해 **Gist**에 올릴 Code를 작성하거나 Gist로 부터 Code를 불러오는 기능 등을 활용할 수 있습니다.

위에서 등록한 `Gist`에 올린 code를 한번 불러오기 위해 다시 `Command+Shift+p`를 누르고 `Gist:Insert Text From Gist File` 이용하면 다음과 같이 리스트에서 Code Snippet을 선택하여 가져올 수 있습니다.

![](/assets/img/2019-03-10-gist-code-snippet/gist_list.png)

![](/assets/img/2019-03-10-gist-code-snippet/insert_from_gist.png)
