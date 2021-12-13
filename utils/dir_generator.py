import datetime
import os
from typing import List, Union

class PathList(object):
    
    def __init__(self, name:str, desc:str='', next=None) -> None:
        super().__init__()
        self.name = name
        self.desc = desc
        self.next = next
    
    def __repr__(self) -> str:
        fmt_str = ""
        fmt_str += f"{self.name}({self.desc}) -> "
        if self.next is not None:
            fmt_str += self.next.__repr__() + '\n'
        else:
            fmt_str += 'End.'
        return fmt_str

PathListType = List[Union[str, PathList]]

class DirGenerator(object):
    """Direction Generator. Allow you can use it to create folders easily.
    Usage:
        inputs: "a.b.c a.b.e f.d"
        dir_gen = DirGenerator(root=args.root, email=args.email, author=args.author)
        dir_gen.makedirs(args.pathlist)
        
        output:     
            |- a
            |-|- b
            |-|-|- c
            |-|-|- e
            |- f
            |-|- d
    """
    def __init__(self, root:str, email:str, author:str) -> None:
        super().__init__()

        self._root = root
        self._email = email
        self._author = author

    @property
    def author(self) -> str:
        return self._author
    @author.setter
    def set_author(self, author:str):
        self._author = author
    
    @property
    def root(self) -> str:
        return self._root
    @root.setter
    def set_root(self, root:str):
        self._root = root

    @property
    def email(self) -> str:
        return self._email
    @email.setter
    def set_email(self, email:str):
        self._email = email
 
    def docstrings(self, desc:str) -> str:
        """create docsrtings for new dir.

        Args:
            desc (str): description for the new dir.

        Returns:
            str: docstrings.
        """
        fmt_str = "#!/usr/bin/env python\n"
        fmt_str += "# -*- coding: utf-8 -*-\n"
        fmt_str += f"# @Author : {self._author} ({self._email})\n"
        fmt_str += f"# @Desc   : {desc}\n"
        fmt_str += f"# @Date   : {datetime.date.today().isoformat()}"
        return fmt_str

    @staticmethod
    def trans_params_to_path_list(path_list:List[str], root:str='./') -> List[PathList]:
        """transform the input params to PathList.

        Args:
            path_list (List[str]): [description]
            root (str, optional): [description]. Defaults to './'.

        Returns:
            List[PathList]: [description]
        """
        def _trans(n, x, i):
            if i == len(x): return n
            n.next = _trans(PathList(x[i]), x, i+1)
            return n

        to_make_path_list = []
        for path in path_list:
            to_make_path_list.append(_trans(PathList(root), path.split('.'), 0))
        return to_make_path_list

    def makedirs(self, path_list:PathListType) -> None:
        """make directions by a list of path.

        Args:
            path_list (List[Union[str, PathList]]): List of input params or PathList.
        """
        if len(path_list) == 0: return

        if not isinstance(path_list[0], PathList):
            to_make_path_list = self.trans_params_to_path_list(path_list, self._root)

        for path_list in to_make_path_list:
            to_make = []
            while path_list:
                to_make.append(path_list.name)
            
                path = os.path.join(*to_make)            
                if not os.path.exists(path):
                    os.makedirs(path)
                    print(f'making dirs in {os.path.abspath(path)}.')
                else:
                    path_list = path_list.next
                    continue

                with open(os.path.join(path, '__init__.py'), 'w', encoding='utf-8') as f:
                    f.write(self.docstrings(path_list.desc))

                path_list = path_list.next
            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./', help='')
    parser.add_argument('--email', type=str, default='chenzejian19@email.szu.edu.cn', help='')
    parser.add_argument('--author', type=str, default='CchenzJ', help='')
    parser.add_argument('--pathlist', type=str, nargs='+', help='')
    
    args = parser.parse_args()
    print(args)

    dir_gen = DirGenerator(root=args.root, email=args.email, author=args.author)
    dir_gen.makedirs(args.pathlist)
    