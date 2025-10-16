DEPS_PREFIX="${HOME}/miopen-deps"

echo "Creating build directories..."
mkdir -p build
mkdir -p "${DEPS_PREFIX}"

echo "Installing dependencies to ${DEPS_PREFIX}..."
cmake -P install_deps.cmake --prefix "${DEPS_PREFIX}"