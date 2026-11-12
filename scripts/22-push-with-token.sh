#!/usr/bin/env bash
# ==============================================================================
# 推送 vision-benchmark 到 GitHub（需要一次 GitHub Personal Access Token）
#
# 背景：本机 github.com 被 DNS 解析为 ::1 导致直连被拒，必须经 mihomo 代理
#      （已写入仓库级 git 配置 http.https://github.com.proxy）。因此凭据只需
#      配置一次，之后 git push 即可直接使用。
#
# 用法：
#   GitHub 新建 token（勾 repo 或 Contents: Read and write），然后
#     read -rsp '粘贴 token（输入不回显）: ' GH_TOKEN; echo
#     export GH_TOKEN
#     bash scripts/22-push-with-token.sh
#
# 安全性：token 只经 stdin 交给 git credential approve，不进入命令行参数、
#        不进入 shell 历史、不写入仓库；凭据文件权限设为 600。
#        校验失败会自动回滚凭据，避免无效 token 影响后续访问。
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

USER_NAME="lcw918110"
HOST="github.com"
PROXY="http://127.0.0.1:17890"

if [[ -z "${GH_TOKEN:-}" ]]; then
  echo "错误：未设置 GH_TOKEN。" >&2
  echo "请先执行：read -rsp '粘贴 token: ' GH_TOKEN; echo; export GH_TOKEN" >&2
  exit 1
fi

# 1. 代理必须可用，否则 DNS 会把 github.com 解析成 ::1
if ! (exec 3<>/dev/tcp/127.0.0.1/17890) 2>/dev/null; then
  echo "错误：代理 127.0.0.1:17890 不可用，无法访问 github.com。" >&2
  exit 1
fi
git config --local "http.https://${HOST}.proxy" "$PROXY"

# 2. 记录凭据（stdin 传入，避免出现在进程列表与历史里）
#    git credential approve 需要 helper 才能真正落盘，store 会写入 ~/.git-credentials
git config --local credential.helper store
printf 'protocol=https\nhost=%s\nusername=%s\npassword=%s\n\n' \
  "$HOST" "$USER_NAME" "$GH_TOKEN" | git credential approve
chmod 600 "${HOME}/.git-credentials" 2>/dev/null || true

# 2.1 确认凭据确实落盘（否则推送会退回交互式提示而失败）
if ! grep -qF -- "$GH_TOKEN" "${HOME}/.git-credentials" 2>/dev/null; then
  echo "错误：凭据未能写入 ${HOME}/.git-credentials。" >&2
  exit 1
fi

# 2.2 确认凭据没有落进仓库配置（.git/config 会随仓库被复制/分享）
if git config --list --local | grep -qF -- "$GH_TOKEN"; then
  echo "警告：凭据疑似写入仓库配置，请检查 .git/config 并移除。" >&2
fi
unset GH_TOKEN

# 3. 校验写权限。注意不能用 ls-remote：本仓库是公开的，匿名也能读成功，
#    那样会把无效 token 误判为有效。只有 dry-run 推送才真正验证写权限。
if ! git push --dry-run --porcelain origin main >/dev/null 2>&1; then
  echo "错误：凭据无效或无权写入该仓库，已回滚本地凭据配置。" >&2
  # 无效凭据留在 ~/.git-credentials 会污染后续所有 github.com 访问，必须清掉
  rm -f "${HOME}/.git-credentials"
  git config --local --unset credential.helper 2>/dev/null || true
  exit 1
fi

# 4. 推送
echo "待推送提交："
git log --oneline "origin/main..HEAD" || true
git push origin main
echo "推送完成："
git log --oneline -1