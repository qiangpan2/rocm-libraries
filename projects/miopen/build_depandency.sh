apt install -y libboost-tools-dev

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
source ~/.bashrc

DEPS_PREFIX="${HOME}/miopen-deps"
mkdir -p build
mkdir -p "${DEPS_PREFIX}"

echo "Installing dependencies to ${DEPS_PREFIX}..."
cmake -P install_deps.cmake --prefix "${DEPS_PREFIX}"