// 切换按钮状态的函数
function toggleButtonState(button, isLoading) {
    if (isLoading) {
        button.textContent = '请求中...';
        button.disabled = true;
    } else {
        button.textContent = '发送';
        button.disabled = false;
    }
}


// 加载删除索引列表
async function loadDeleteIndexList() {
    try {
        const response = await fetch('http://localhost:8001/get-index-names', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        
        const data = await response.json();
        const deleteIndexList = document.getElementById('deleteIndexList');
        
        if (data.status === 'success' && data.index_names.length > 0) {
            deleteIndexList.innerHTML = data.index_names.map(name => 
                `<div class="index-item">
                    <input type="checkbox" id="delete-${name}" value="${name}">
                    <label for="delete-${name}">${name}</label>
                </div>`
            ).join('');
        } else {
            deleteIndexList.innerHTML = '<div class="no-data">没有可用的索引</div>';
        }
    } catch (error) {
        console.error('Error loading delete index list:', error);
        document.getElementById('deleteIndexList').innerHTML = '<div class="error">加载索引列表失败</div>';
    }
}

// 获取相关元素
const indexFilter = document.getElementById('indexFilter');
const indexList = document.getElementById('indexList');

// 添加输入事件监听器
indexFilter.addEventListener('input', function() {
    const filterText = this.value.toLowerCase();
    const indexItems = indexList.querySelectorAll('.index-item');
    
    indexItems.forEach(item => {
        const indexName = item.textContent.toLowerCase();
        if (indexName.includes(filterText)) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
});

// 加载索引名称
async function loadIndexNames() {
    try {
        const response = await fetch('http://localhost:8001/get-index-names', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        
        const data = await response.json();
        const indexList = document.getElementById('indexList');
        
        if (data.status === 'success' && data.index_names.length > 0) {
            indexList.innerHTML = data.index_names.map(name => 
                `<div class="index-item">${name}</div>`
            ).join('');
            
            // Add click event to index items
            document.querySelectorAll('.index-item').forEach(item => {
                item.addEventListener('click', function() {
                    // 切换选中状态
                    this.classList.toggle('selected');
                });
            });
        } else {
            indexList.innerHTML = '<div class="no-data">没有可用的索引</div>';
        }
    } catch (error) {
        console.error('Error loading index names:', error);
        document.getElementById('indexList').innerHTML = '<div class="error">加载索引列表失败</div>';
    }
}

// 初始化事件监听器
function initEventListeners() {
    const modal = document.getElementById('indexManagementPanel');
    const closeBtn = document.querySelector('.close');
    const manageBtn = document.getElementById('manageIndexBtn');
    const deleteIndexBtn = document.getElementById('deleteIndexBtn');
    const addIndexBtn = document.getElementById('addIndexBtn');
    const selectFilesBtn = document.getElementById('selectIndividualFilesBtn');
    let selectedFiles = [];

    // 打开管理面板
    manageBtn.addEventListener('click', async function() {
        modal.style.display = 'block';
        await loadDeleteIndexList();
    });

    // 关闭管理面板
    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });

    // 删除索引
    deleteIndexBtn.addEventListener('click', async function() {
        const selectedIndexes = Array.from(document.querySelectorAll('#deleteIndexList input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);

        if (selectedIndexes.length === 0) {
            alert('请至少选择一个要删除的索引！');
            return;
        }

        try {
            const response = await fetch('http://localhost:8001/delete-index', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    index_names: selectedIndexes
                })
            });

            const result = await response.json();
            if (result.status === 'success') {
                alert('索引删除成功！');
                await loadIndexNames();
                await loadDeleteIndexList();
            } else {
                alert('索引删除失败！');
            }
        } catch (error) {
            console.error('Error deleting index:', error);
            alert(`删除索引过程中发生错误：${error.message}`);
        }
    });

    // 选择文件
    selectFilesBtn.addEventListener('click', function() {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = true;

        input.addEventListener('change', function() {
            selectedFiles = Array.from(this.files).map(file => file.path);
            if (selectedFiles.length > 0) {
                document.getElementById('selectedFilesInfo').textContent = `已选择 ${selectedFiles.length} 个文件`;
            } else {
                document.getElementById('selectedFilesInfo').textContent = '未选择任何文件';
            }
        });
        
        input.click();
    });

    // 添加索引
    addIndexBtn.addEventListener('click', async function() {
        const originalText = this.textContent;
        this.textContent = '添加中...';
        this.disabled = true;

        const requestBody = {
            input_path: selectedFiles.length === 1 ? selectedFiles[0] : selectedFiles,
            index_name: document.getElementById('indexName').value,
            index_type: document.getElementById('indexType').value,
            chunk_size: parseInt(document.getElementById('chunkSize').value),
            chunk_overlap: parseInt(document.getElementById('chunkOverlap').value),
            file_extension: document.getElementById('fileExtension').value || null
        };

        try {
            const response = await fetch('http://localhost:8001/build-index', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Error details:', errorData);
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.status === 'success') {
                alert('索引添加成功！');
                selectedFiles = [];
                document.getElementById('selectedFilesInfo').textContent = '';
                await loadIndexNames();
                await loadDeleteIndexList();
            } else {
                alert('索引添加失败！');
            }
        } catch (error) {
            console.error('Error adding index:', error);
            alert(`添加索引过程中发生错误：${error.message}`);
        } finally {
            this.textContent = originalText;
            this.disabled = false;
        }
    });

    // 修改流式输出部分
    let currentContent = ''; // 用于存储当前 content
    function updateStreamOutput(content, question) {
        const outputElement = document.getElementById('streamOutput');
        
        // 将问题和回答内容拼接
        const fullContent = question ? `提示词：${question}\n\n${content}` : content;
        currentContent = fullContent; // 更新当前 content
        // 将内容按<think>和</think>分割
        const parts = fullContent.split(/(<think>|<\/think>)/g);
        
        let formattedContent = '';
        let isThinking = false;
        
        parts.forEach(part => {
            if (part === '<think>') {
                isThinking = true;
                formattedContent += `<div class="thinking-content">`;
            } else if (part === '</think>') {
                isThinking = false;
                formattedContent += `</div>`;
            } else {
                if (isThinking) {
                    formattedContent += part;
                } else {
                    formattedContent += `<div>${part}</div>`;
                }
            }
        });
        
        outputElement.innerHTML = marked.parse(formattedContent);
        outputElement.scrollTop = outputElement.scrollHeight;
    }

    // 复制结果到剪贴板
    document.getElementById('copyButton').addEventListener('click', function() {
        if (currentContent) {
            navigator.clipboard.writeText(currentContent)
                .then(function() {
                    alert('已复制到剪贴板');
                })
                .catch(function(error) {
                    alert('复制失败: ' + error);
                });
        } else {
            alert('没有内容可复制');
        }
    });

    // 新增对话历史管理功能
    const CHAT_HISTORY_KEY = 'chat_history';
    const MAX_HISTORY_LENGTH = 10;

    // 获取对话历史
    function getChatHistory() {
        const history = localStorage.getItem(CHAT_HISTORY_KEY);
        return history ? JSON.parse(history) : [];
    }

    // 保存对话
    function saveChatMessage(role, content) {
        const history = getChatHistory();
        history.push({ role, content, timestamp: new Date().toISOString() });
        if (history.length > MAX_HISTORY_LENGTH) {
            history.shift(); // 移除最旧的消息
        }
        localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history));
    }

    // 格式化历史记录为上下文
    function formatHistoryContext() {
        const history = getChatHistory();
        return history.map(msg => {
            const prefix = msg.role === 'user' ? 'User: ' : 'AI: ';
            return prefix + msg.content;
        }).join('\n');
    }

    // 查询表单提交
    document.getElementById('queryForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const button = document.getElementById('queryButton');
        toggleButtonState(button, true);
        
        const question = document.getElementById('question').value;
        const similarityTopK = parseInt(document.getElementById('similarityTopK').value) || 12;
        const selectedIndexes = Array.from(document.querySelectorAll('.index-item.selected'))
            .map(item => item.textContent);

        if (selectedIndexes.length === 0) {
            alert('请至少选择一个索引！');
            toggleButtonState(button, false);
            return;
        }

        document.getElementById('streamOutput').textContent = '';
        
        try {
            const response = await fetch('http://localhost:8001/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    index_names: selectedIndexes,
                    similarity_top_k: similarityTopK
                })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                result += chunk;
                
                updateStreamOutput(result, question);
                
                const outputElement = document.getElementById('streamOutput');
                outputElement.scrollTop = outputElement.scrollHeight;
            }
            
        } catch (error) {
            document.getElementById('streamOutput').textContent = `Error: ${error.message}`;
        } finally {
            toggleButtonState(button, false);
        }
    });

    // 修改聊天表单提交逻辑
    document.getElementById('chatForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const button = document.getElementById('chatButton');
        toggleButtonState(button, true);
        
        const question = document.getElementById('chatQuestion').value;
        const context = formatHistoryContext();

        document.getElementById('streamOutput').textContent = '';
        
        try {
            const response = await fetch('http://localhost:8001/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    context: context
                })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                result += chunk;
                
                updateStreamOutput(result, question);
            }
            
            // 保存对话历史
            saveChatMessage('user', question);
            saveChatMessage('ai', result);
            
        } catch (error) {
            document.getElementById('streamOutput').textContent = `Error: ${error.message}`;
        } finally {
            toggleButtonState(button, false);
        }
    });

    // 添加清除历史记录的事件监听
    document.getElementById('clearHistoryButton').addEventListener('click', function() {
        localStorage.removeItem(CHAT_HISTORY_KEY);
        document.getElementById('streamOutput').textContent = '';
        alert('历史记录已清除');
    });
}

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initEventListeners();
    loadIndexNames();
});