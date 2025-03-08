// main.js - 重构版本

// 模块化设计 - 将功能分为不同模块
// ==========================================================

// API 服务模块 - 处理所有与后端的通信
const ApiService = {
    // 基础 URL
    baseUrl: 'http://localhost:8001',

    // 通用 fetch 方法
    async fetchData(endpoint, data = {}, options = {}) {
        const url = `${this.baseUrl}/${endpoint}`;
        const defaultOptions = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        };
        
        const fetchOptions = { ...defaultOptions, ...options };
        return fetch(url, fetchOptions);
    },

    // 获取索引名称列表
    async getIndexNames() {
        try {
            const response = await this.fetchData('get-index-names', {});
            return await response.json();
        } catch (error) {
            console.error('Error fetching index names:', error);
            throw error;
        }
    },

    // 删除索引
    async deleteIndex(indexNames) {
        try {
            const response = await this.fetchData('delete-index', {
                index_names: indexNames
            });
            return await response.json();
        } catch (error) {
            console.error('Error deleting index:', error);
            throw error;
        }
    },

    // 构建索引
    async buildIndex(indexData) {
        try {
            const response = await this.fetchData('build-index', indexData);
            if (!response.ok) {
                const errorData = await response.json();
                console.error('Error details:', errorData);
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error building index:', error);
            throw error;
        }
    },

    // 查询
    async query(question, indexNames, similarityTopK, signal) {
        try {
            const response = await this.fetchData('query', {
                question: question,
                index_names: indexNames,
                similarity_top_k: similarityTopK
            }, { signal });
            
            return response;
        } catch (error) {
            console.error('Query error:', error);
            throw error;
        }
    },

    // 聊天
    async chat(question, context, signal) {
        try {
            const response = await this.fetchData('chat', {
                question: question,
                context: context
            }, { signal });
            
            return response;
        } catch (error) {
            console.error('Chat error:', error);
            throw error;
        }
    }
};

// UI 工具模块 - 处理通用 UI 操作
const UiUtils = {
    // 切换按钮状态
    toggleButtonState(button, isLoading) {
        if (isLoading) {
            button.textContent = '请求中...';
            button.disabled = true;
        } else {
            button.textContent = '发送';
            button.disabled = false;
        }
    },

    // 切换终止按钮状态
    toggleTerminateButtonState(button, isEnabled) {
        button.disabled = !isEnabled;
    },

    // 显示提示信息
    showAlert(message) {
        alert(message);
    },

    // 更新元素内容
    updateElementContent(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = content;
        }
    },

    // 获取选中的索引
    getSelectedIndexes() {
        return Array.from(document.querySelectorAll('.index-item.selected'))
            .map(item => item.textContent);
    },
    
    // 更新已选择的索引列表
    updateSelectedIndexesList() {
        const selectedIndexes = this.getSelectedIndexes();
        const selectedIndexList = document.getElementById('selectedIndexList');
        
        // 清空当前内容
        selectedIndexList.innerHTML = '';
        
        if (selectedIndexes.length === 0) {
            // 如果没有选中的索引，显示提示信息
            const noSelection = document.createElement('div');
            noSelection.className = 'no-selection';
            noSelection.textContent = '未选择任何索引';
            selectedIndexList.appendChild(noSelection);
        } else {
            // 为每个选中的索引创建一个元素
            selectedIndexes.forEach(indexName => {
                const indexItem = document.createElement('div');
                indexItem.className = 'selected-index-item';
                
                const indexText = document.createElement('span');
                indexText.textContent = indexName;
                
                const removeBtn = document.createElement('span');
                removeBtn.className = 'remove-btn';
                removeBtn.textContent = '×';
                removeBtn.setAttribute('title', '移除此索引');
                
                // 点击移除按钮时，在主列表中找到对应索引并模拟点击
                removeBtn.addEventListener('click', function() {
                    const indexListItems = document.querySelectorAll('#indexList .index-item');
                    indexListItems.forEach(item => {
                        if (item.textContent === indexName && item.classList.contains('selected')) {
                            item.click(); // 这将切换主列表中的选择状态
                        }
                    });
                });
                
                indexItem.appendChild(indexText);
                indexItem.appendChild(removeBtn);
                selectedIndexList.appendChild(indexItem);
            });
        }
    },

    // 获取选中的要删除的索引
    getSelectedIndexesToDelete() {
        return Array.from(document.querySelectorAll('#deleteIndexList input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);
    },

    // 节流函数 - 限制函数调用频率
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
};

// 流式输出处理模块
const StreamProcessor = {
    currentContent: '',
    lastProcessedLength: 0,
    
    // 更新流式输出
    updateStreamOutput(content, question) {
        const outputElement = document.getElementById('streamOutput');
        
        // 将问题和回答内容拼接
        const fullContent = question ? `提示词：${question}\n\n${content}` : content;
        this.currentContent = fullContent;
        
        // 只处理新增的内容，避免重复处理整个内容
        if (fullContent.length > this.lastProcessedLength) {
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
            
            // 使用requestAnimationFrame优化渲染性能
            requestAnimationFrame(() => {
                outputElement.innerHTML = marked.parse(formattedContent);
                outputElement.scrollTop = outputElement.scrollHeight;
            });
            
            this.lastProcessedLength = fullContent.length;
        }
    },
    
    // 重置处理状态
    reset() {
        this.currentContent = '';
        this.lastProcessedLength = 0;
        UiUtils.updateElementContent('streamOutput', '');
    },
    
    // 显示加载提示
    showLoading() {
        UiUtils.updateElementContent('streamOutput', '<div class="loading">正在处理，请稍候...</div>');
    },
    
    // 显示错误信息
    showError(message) {
        UiUtils.updateElementContent('streamOutput', `<div class="error">${message}</div>`);
    },
    
    // 添加错误信息（不清除现有内容）
    appendError(message) {
        const outputElement = document.getElementById('streamOutput');
        outputElement.innerHTML += `<div class="error">${message}</div>`;
    }
};

// 本地存储模块 - 处理聊天历史
const StorageManager = {
    CHAT_HISTORY_KEY: 'chat_history',
    MAX_HISTORY_LENGTH: 10,
    
    // 获取聊天历史
    getChatHistory() {
        const history = localStorage.getItem(this.CHAT_HISTORY_KEY);
        return history ? JSON.parse(history) : [];
    },
    
    // 保存聊天消息
    saveChatMessage(role, content) {
        const history = this.getChatHistory();
        history.push({ role, content, timestamp: new Date().toISOString() });
        if (history.length > this.MAX_HISTORY_LENGTH) {
            history.shift(); // 移除最旧的消息
        }
        localStorage.setItem(this.CHAT_HISTORY_KEY, JSON.stringify(history));
    },
    
    // 格式化历史记录为上下文
    formatHistoryContext() {
        const history = this.getChatHistory();
        return history.map(msg => {
            const prefix = msg.role === 'user' ? 'User: ' : 'AI: ';
            return prefix + msg.content;
        }).join('\n');
    },
    
    // 清除历史记录
    clearHistory() {
        localStorage.removeItem(this.CHAT_HISTORY_KEY);
    }
};

// 请求控制器模块 - 管理请求的中止控制器
const RequestController = {
    queryAbortController: null,
    chatAbortController: null,
    
    // 创建新的查询控制器
    createQueryController() {
        if (this.queryAbortController) {
            this.abortQuery();
        }
        this.queryAbortController = new AbortController();
        return this.queryAbortController.signal;
    },
    
    // 中止查询
    abortQuery() {
        if (this.queryAbortController) {
            this.queryAbortController.abort();
            this.queryAbortController = null;
        }
    },
    
    // 创建新的聊天控制器
    createChatController() {
        if (this.chatAbortController) {
            this.abortChat();
        }
        this.chatAbortController = new AbortController();
        return this.chatAbortController.signal;
    },
    
    // 中止聊天
    abortChat() {
        if (this.chatAbortController) {
            this.chatAbortController.abort();
            this.chatAbortController = null;
        }
    }
};

// 索引管理模块 - 处理索引相关操作
const IndexManager = {
    selectedFiles: [],
    
    // 加载索引名称列表
    async loadIndexNames() {
        try {
            const data = await ApiService.getIndexNames();
            const indexList = document.getElementById('indexList');
            
            if (data.status === 'success' && data.index_names.length > 0) {
                indexList.innerHTML = data.index_names.map(name => 
                    `<div class="index-item">${name}</div>`
                ).join('');
                
                // 添加点击事件
                document.querySelectorAll('.index-item').forEach(item => {
                item.addEventListener('click', function() {
                    // 切换选中状态
                    this.classList.toggle('selected');
                    // 更新已选择的索引列表
                    UiUtils.updateSelectedIndexesList();
                });
                });
            } else {
                indexList.innerHTML = '<div class="no-data">没有可用的索引</div>';
            }
        } catch (error) {
            console.error('Error loading index names:', error);
            document.getElementById('indexList').innerHTML = '<div class="error">加载索引列表失败</div>';
        }
    },
    
    // 加载删除索引列表
    async loadDeleteIndexList() {
        try {
            const data = await ApiService.getIndexNames();
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
    },
    
    // 删除选中的索引
    async deleteSelectedIndexes() {
        const selectedIndexes = UiUtils.getSelectedIndexesToDelete();
        
        if (selectedIndexes.length === 0) {
            UiUtils.showAlert('请至少选择一个要删除的索引！');
            return;
        }
        
        try {
            const result = await ApiService.deleteIndex(selectedIndexes);
            if (result.status === 'success') {
                UiUtils.showAlert('索引删除成功！');
                await this.loadIndexNames();
                await this.loadDeleteIndexList();
            } else {
                UiUtils.showAlert('索引删除失败！');
            }
        } catch (error) {
            console.error('Error deleting index:', error);
            UiUtils.showAlert(`删除索引过程中发生错误：${error.message}`);
        }
    },
    
    // 添加索引
    async addIndex(button) {
        const originalText = button.textContent;
        button.textContent = '添加中...';
        button.disabled = true;
        
        const requestBody = {
            input_path: this.selectedFiles.length === 1 ? this.selectedFiles[0] : this.selectedFiles,
            index_name: document.getElementById('indexName').value,
            index_type: document.getElementById('indexType').value,
            chunk_size: parseInt(document.getElementById('chunkSize').value),
            chunk_overlap: parseInt(document.getElementById('chunkOverlap').value),
            file_extension: document.getElementById('fileExtension').value || null
        };
        
        try {
            const result = await ApiService.buildIndex(requestBody);
            if (result.status === 'success') {
                UiUtils.showAlert('索引添加成功！');
                this.selectedFiles = [];
                document.getElementById('selectedFilesInfo').textContent = '';
                await this.loadIndexNames();
                await this.loadDeleteIndexList();
            } else {
                UiUtils.showAlert('索引添加失败！');
            }
        } catch (error) {
            console.error('Error adding index:', error);
            UiUtils.showAlert(`添加索引过程中发生错误：${error.message}`);
        } finally {
            button.textContent = originalText;
            button.disabled = false;
        }
    },
    
    // 选择文件
    selectFiles() {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = true;
        
        input.addEventListener('change', () => {
            this.selectedFiles = Array.from(input.files).map(file => file.path);
            if (this.selectedFiles.length > 0) {
                document.getElementById('selectedFilesInfo').textContent = `已选择 ${this.selectedFiles.length} 个文件`;
            } else {
                document.getElementById('selectedFilesInfo').textContent = '未选择任何文件';
            }
        });
        
        input.click();
    },
    
    // 过滤索引列表
    filterIndexList(filterText) {
        const indexItems = document.querySelectorAll('#indexList .index-item');
        const normalizedFilter = filterText.toLowerCase();
        
        indexItems.forEach(item => {
            const indexName = item.textContent.toLowerCase();
            item.style.display = indexName.includes(normalizedFilter) ? 'block' : 'none';
        });
    },
    
    // 过滤删除索引列表
    filterDeleteIndexList(filterText) {
        const indexItems = document.querySelectorAll('#deleteIndexList .index-item');
        const normalizedFilter = filterText.toLowerCase();
        
        indexItems.forEach(item => {
            const indexName = item.querySelector('label').textContent.toLowerCase();
            item.style.display = indexName.includes(normalizedFilter) ? 'block' : 'none';
        });
    }
};

// 查询处理模块 - 处理查询相关操作
const QueryProcessor = {
    // 处理查询提交
    async handleQuerySubmit(event) {
        event.preventDefault();
        
        // 确保之前的请求被清理
        await this.ensureCleanup();
        
        const button = document.getElementById('queryButton');
        const terminateButton = document.getElementById('queryTerminateButton');
        UiUtils.toggleButtonState(button, true);
        UiUtils.toggleTerminateButtonState(terminateButton, true);
        
        const question = document.getElementById('question').value;
        const similarityTopK = parseInt(document.getElementById('similarityTopK').value) || 10;
        const selectedIndexes = UiUtils.getSelectedIndexes();
        
        if (selectedIndexes.length === 0) {
            UiUtils.showAlert('请至少选择一个索引！');
            UiUtils.toggleButtonState(button, false);
            UiUtils.toggleTerminateButtonState(terminateButton, false);
            return;
        }
        
        // 重置输出区域
        StreamProcessor.reset();
        StreamProcessor.showLoading();
        
        // 创建新的 AbortController
        const signal = RequestController.createQueryController();
        
        let reader = null;
        
        try {
            const response = await ApiService.query(question, selectedIndexes, similarityTopK, signal);
            
            // 清除加载提示
            StreamProcessor.reset();
            
            reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';
            
            // 使用节流函数来限制更新频率
            const throttledUpdate = UiUtils.throttle((content, question) => {
                StreamProcessor.updateStreamOutput(content, question);
            }, 50);
            
            // 添加信号监听器，在请求被终止时释放reader
            signal.addEventListener('abort', async () => {
                if (reader) {
                    try {
                        await reader.cancel();
                        reader = null;
                    } catch (e) {
                        console.error('Error cancelling reader:', e);
                    }
                }
            });
            
            while (!signal.aborted) {
                try {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, { stream: true });
                    result += chunk;
                    
                    throttledUpdate(result, question);
                } catch (e) {
                    if (signal.aborted) break;
                    throw e;
                }
            }
            
            // 确保最后一次更新显示完整内容
            if (!signal.aborted) {
                StreamProcessor.updateStreamOutput(result, question);
            }
            
        } catch (error) {
            if (error.name === 'AbortError') {
                StreamProcessor.appendError('请求已终止');
            } else {
                StreamProcessor.showError(`Error: ${error.message}`);
            }
        } finally {
            // 确保reader被释放
            if (reader) {
                try {
                    await reader.cancel();
                } catch (e) {
                    console.error('Error cancelling reader in finally block:', e);
                }
                reader = null;
            }
            
            UiUtils.toggleButtonState(button, false);
            UiUtils.toggleTerminateButtonState(terminateButton, false);
            RequestController.queryAbortController = null;
        }
    },
    
    // 确保之前的请求被清理
    async ensureCleanup() {
        if (RequestController.queryAbortController) {
            RequestController.abortQuery();
            // 添加小延迟，确保连接完全关闭
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    },
    
    // 终止查询
    terminateQuery() {
        if (RequestController.queryAbortController) {
            RequestController.abortQuery();
            document.getElementById('queryTerminateButton').disabled = true;
        }
    }
};

// 聊天处理模块 - 处理聊天相关操作
const ChatProcessor = {
    // 处理聊天提交
    async handleChatSubmit(event) {
        event.preventDefault();
        
        // 确保之前的请求被清理
        await this.ensureCleanup();
        
        const button = document.getElementById('chatButton');
        const terminateButton = document.getElementById('chatTerminateButton');
        UiUtils.toggleButtonState(button, true);
        UiUtils.toggleTerminateButtonState(terminateButton, true);
        
        const question = document.getElementById('chatQuestion').value;
        const context = StorageManager.formatHistoryContext();
        
        // 重置输出区域
        StreamProcessor.reset();
        StreamProcessor.showLoading();
        
        // 创建新的 AbortController
        const signal = RequestController.createChatController();
        
        let reader = null;
        
        try {
            const response = await ApiService.chat(question, context, signal);
            
            // 清除加载提示
            StreamProcessor.reset();
            
            reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';
            
            // 使用节流函数来限制更新频率
            const throttledUpdate = UiUtils.throttle((content, question) => {
                StreamProcessor.updateStreamOutput(content, question);
            }, 50);
            
            // 添加信号监听器，在请求被终止时释放reader
            signal.addEventListener('abort', async () => {
                if (reader) {
                    try {
                        await reader.cancel();
                        reader = null;
                    } catch (e) {
                        console.error('Error cancelling reader:', e);
                    }
                }
            });
            
            while (!signal.aborted) {
                try {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, { stream: true });
                    result += chunk;
                    
                    throttledUpdate(result, question);
                } catch (e) {
                    if (signal.aborted) break;
                    throw e;
                }
            }
            
            // 确保最后一次更新显示完整内容
            if (!signal.aborted) {
                StreamProcessor.updateStreamOutput(result, question);
                
                // 保存对话历史
                StorageManager.saveChatMessage('user', question);
                StorageManager.saveChatMessage('ai', result);
            }
            
        } catch (error) {
            if (error.name === 'AbortError') {
                StreamProcessor.appendError('请求已终止');
            } else {
                StreamProcessor.showError(`Error: ${error.message}`);
            }
        } finally {
            // 确保reader被释放
            if (reader) {
                try {
                    await reader.cancel();
                } catch (e) {
                    console.error('Error cancelling reader in finally block:', e);
                }
                reader = null;
            }
            
            UiUtils.toggleButtonState(button, false);
            UiUtils.toggleTerminateButtonState(terminateButton, false);
            RequestController.chatAbortController = null;
        }
    },
    
    // 确保之前的请求被清理
    async ensureCleanup() {
        if (RequestController.chatAbortController) {
            RequestController.abortChat();
            // 添加小延迟，确保连接完全关闭
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    },
    
    // 终止聊天
    terminateChat() {
        if (RequestController.chatAbortController) {
            RequestController.abortChat();
            document.getElementById('chatTerminateButton').disabled = true;
        }
    },
    
    // 清除历史记录
    clearHistory() {
        StorageManager.clearHistory();
        StreamProcessor.reset();
        UiUtils.showAlert('历史记录已清除');
    }
};

// 事件处理模块 - 初始化所有事件监听器
const EventHandler = {
    // 初始化所有事件监听器
    initEventListeners() {
        // 索引管理面板相关事件
        this.initIndexManagementEvents();
        
        // 索引过滤相关事件
        this.initIndexFilterEvents();
        
        // 查询相关事件
        this.initQueryEvents();
        
        // 聊天相关事件
        this.initChatEvents();
        
        // 复制结果相关事件
        this.initCopyEvents();
    },
    
    // 初始化索引管理面板相关事件
    initIndexManagementEvents() {
        const modal = document.getElementById('indexManagementPanel');
        const closeBtn = document.querySelector('.close');
        const manageBtn = document.getElementById('manageIndexBtn');
        const deleteIndexBtn = document.getElementById('deleteIndexBtn');
        const addIndexBtn = document.getElementById('addIndexBtn');
        const selectFilesBtn = document.getElementById('selectIndividualFilesBtn');
        
        // 打开管理面板
        manageBtn.addEventListener('click', async function() {
            modal.style.display = 'block';
            await IndexManager.loadDeleteIndexList();
        });
        
        // 关闭管理面板
        closeBtn.addEventListener('click', function() {
            modal.style.display = 'none';
        });
        
        // 删除索引
        deleteIndexBtn.addEventListener('click', () => IndexManager.deleteSelectedIndexes());
        
        // 选择文件
        selectFilesBtn.addEventListener('click', () => IndexManager.selectFiles());
        
        // 添加索引
        addIndexBtn.addEventListener('click', function() {
            IndexManager.addIndex(this);
        });
    },
    
    // 初始化索引过滤相关事件
    initIndexFilterEvents() {
        const indexFilter = document.getElementById('indexFilter');
        const deleteIndexFilter = document.getElementById('deleteIndexFilter');
        
        // 添加主页面索引过滤输入事件监听器
        indexFilter.addEventListener('input', function() {
            IndexManager.filterIndexList(this.value);
        });
        
        // 添加删除索引页面过滤输入事件监听器
        deleteIndexFilter.addEventListener('input', function() {
            IndexManager.filterDeleteIndexList(this.value);
        });
    },
    
    // 初始化查询相关事件
    initQueryEvents() {
        // 查询表单提交
        document.getElementById('queryForm').addEventListener('submit', event => 
            QueryProcessor.handleQuerySubmit(event)
        );
        
        // 查询终止按钮
        document.getElementById('queryTerminateButton').addEventListener('click', () => 
            QueryProcessor.terminateQuery()
        );
    },
    
    // 初始化聊天相关事件
    initChatEvents() {
        // 聊天表单提交
        document.getElementById('chatForm').addEventListener('submit', event => 
            ChatProcessor.handleChatSubmit(event)
        );
        
        // 聊天终止按钮
        document.getElementById('chatTerminateButton').addEventListener('click', () => 
            ChatProcessor.terminateChat()
        );
        
        // 清除历史记录按钮
        document.getElementById('clearHistoryButton').addEventListener('click', () => 
            ChatProcessor.clearHistory()
        );
    },
    
    // 初始化复制结果相关事件
    initCopyEvents() {
        // 复制结果到剪贴板
        document.getElementById('copyButton').addEventListener('click', function() {
            if (StreamProcessor.currentContent) {
                navigator.clipboard.writeText(StreamProcessor.currentContent)
                    .then(() => UiUtils.showAlert('已复制到剪贴板'))
                    .catch(error => UiUtils.showAlert('复制失败: ' + error));
            } else {
                UiUtils.showAlert('没有内容可复制');
            }
        });
    }
};

// 初始化应用
document.addEventListener('DOMContentLoaded', function() {
    // 初始化事件监听器
    EventHandler.initEventListeners();
    
    // 加载索引名称
    IndexManager.loadIndexNames();
    
    // 初始化终止按钮状态
    document.getElementById('queryTerminateButton').disabled = true;
    document.getElementById('chatTerminateButton').disabled = true;
});
