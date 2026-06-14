# Claude Code 使用说明

## 基本原则

本地环境只用于阅读和修改代码，不包含完整运行环境。

不要在本地尝试运行、训练、测试或调试项目代码。任何依赖数据集、Python 环境、CUDA、GPU、Linux 路径或服务器依赖的操作，都应在服务器上完成。

## 目录限制

只允许修改当前项目中的 `workplace/` 目录下的内容。

不要修改 `workplace/` 目录外的任何文件，除非用户明确要求。

## Git 操作限制

不要执行任何 Git 操作。

包括但不限于：

```bash
git add
git commit
git push
git pull
git reset
git checkout
git merge
git rebase
git clean
```

所有 Git 操作必须由用户本人完成。

如果需要提交代码，只能告诉用户修改了哪些文件，并建议用户自行执行 Git 命令。

## 服务器路径说明

服务器端项目主目录为：

```bash
/share/home/tm1078571822880000/a904903640/group7
```

该服务器目录与 Claude Code 当前所在的本地项目目录处于同一项目层级。

本地只负责修改代码；服务器负责运行代码。
