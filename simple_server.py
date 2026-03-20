#!/usr/bin/env python3
"""简单的文件浏览 Web 服务 - 用于远程访问项目文件"""

import http.server
import socketserver
import os
import urllib.parse

PORT = 50080
DIRECTORY = "/workspace/PTX-EMU"

class FileBrowserHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        # 解码 URL
        parsed_path = urllib.parse.urlparse(self.path)
        decoded_path = urllib.parse.unquote(parsed_path.path)
        
        # 安全处理路径
        if decoded_path == '/':
            self.send_response(302)
            self.send_header('Location', '/browse/')
            self.end_headers()
            return
        
        # 调用父类处理
        super().do_GET()
    
    def log_message(self, format, *args):
        # 美化日志输出
        print(f"[{self.address_string()}] {format % args}")

def run_server():
    with socketserver.TCPServer(("", PORT), FileBrowserHandler) as httpd:
        print(f"🌐 文件浏览服务已启动")
        print(f"📁 目录：{DIRECTORY}")
        print(f"🔗 本地访问：http://localhost:{PORT}")
        print(f"🔐 通过 SSH 隧道访问：http://localhost:50080")
        print(f"\n按 Ctrl+C 停止服务")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 服务已停止")

if __name__ == "__main__":
    run_server()
